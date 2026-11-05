"""Hand-constructed one-block multiplier, derived from the supplied prior build.
No training, task-data fitting, or automatic parameter search is used.
Only arch.py is read as an external input when this program runs.
"""

import json
import sys
import math
import importlib.util
from pathlib import Path
import numpy as np

# Exact solutions of the embedding sum/norm equations, in units of 1/8.
# Derived without task data; checked algebraically for every digit below.
PADDING = [[-128, -128, 128, 128], [-219, 35, 102, 77], [-224, 52, 92, 64], [-227, 57, 74, 71], [-228, 36, 84, 68], [-231, 49, 78, 51], [-232, 28, 76, 56], [-231, 11, 74, 57], [-232, 8, 56, 56], [-231, 14, 55, 29]]

def build():
    spec = importlib.util.spec_from_file_location("task_arch", Path(__file__).resolve().parents[1] / "arch.py")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    upper = [41,90,140,190,158,109,59,9]
    diagonal = [1,2,3,4,3,2,1,0]
    centers = [20,40,60,80,60,40,20,0]
    cfg = dict(n_layer=1, loop=1, n_head=24, d_head=4,
               d_model=96, d_mlp=9+2*sum(u//10 for u in upper))
    W = {n: np.zeros(s,np.float32) for n,s in arch.shapes(cfg).items()}
    A = cfg["n_head"]*cfg["d_head"]
    val = lambda k: 56+2*k
    pos = lambda p: val(p-11) if 11 <= p <= 18 else 16+2*p

    for t in range(14):
        if t < 10:
            d=t
            f=d//2
            W["wte"][t,:6] = [d/2,0,d*d/8,0,f/2,0]
            total = -(4*d+d*d+4*f)
            norm = 65536-16*d*d-d**4-16*f*f
            pad = PADDING[d]
            assert sum(pad) == total and sum(x*x for x in pad) == norm
            W["wte"][t,6:10] = np.asarray(pad,np.float32)/8
    for p in [0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,18]:
        if 11 <= p <= 18:
            W["wpe"][p,pos(p)] = -16
            W["wpe"][p,14] = 16
        else:
            W["wpe"][p,pos(p):pos(p)+2] = [16,-16]
    W["lnf.w"][[0,2]] = 8
    # Positive digit-only offset: marker embeddings are all zero.
    W["lnf.b"][6] = -1/1024
    W["h0.ln1.w"][[0,2,4]] = 4
    for p in [0,1,2,3,5,6,7,8,12,13,14,15,16,17,18]:
        W["h0.ln1.w"][pos(p)] = 4

    def head(h, positions, biased=False, squares=False, multiplicity=1):
        off=4*h
        W["h0.attn.bqkv"][off] = 2
        for p in positions:
            W["h0.attn.wqkv"][pos(p),A+off] = 2
        if biased:
            W["h0.attn.wqkv"][0,A+off] = 1/256
        W["h0.attn.wqkv"][0,2*A+off] = 2
        if squares:
            W["h0.attn.wqkv"][0,2*A+off] = 2048*multiplicity
            W["h0.attn.wqkv"][2,2*A+off] = 8*multiplicity
            W["h0.attn.wqkv"][4,2*A+off+1] = 2*multiplicity

    def output(h, component, k, coefficient):
        W["h0.attn.wo"][4*h+component,val(k):val(k)+2] = [-coefficient/64,coefficient/64]

    # One uniform head per diagonal; its values include the pair count.
    for s in range(7):
        pairs = [(i,s-i) for i in range(4) if 0 <= s-i < 4]
        positions = [i for i,j in pairs] + [5+j for i,j in pairs]
        head(s,positions,squares=True,multiplicity=len(pairs))
        output(s,1,s,-20)
        for k in range(s,8):
            output(s,0,k,10.0**(s-k))
    for i in range(4):
        for j in range(4):
            h = 7+4*i+j
            head(h,[i,5+j],biased=True)
            for k in range(i+j,8):
                output(h,0,k,-1024*10.0**(i+j-k))
    # One causally truncated geometric average retrieves the answer prefix.
    off=4*23
    W["h0.attn.bqkv"][off:off+2] = 2
    for j in range(7):
        W["h0.attn.wqkv"][pos(12+j),A+off] = -2
        W["h0.attn.wqkv"][pos(12+j),A+off+1] = -j*math.log(10)/16
    W["h0.attn.wqkv"][0,2*A+off] = 2
    for k in range(1,8):
        output(23,0,k,-(1-10.0**(-k))/9)

    for k,u in enumerate(upper):
        W["h0.ln2.w"][val(k)] = 1
        offset = 16 + (centers[k] - 40*diagonal[k])/64
        W["h0.attn.bo"][val(k):val(k)+2] = [offset,-offset]
    W["h0.ln2.w"][14] = 1

    scale,outscale = 4096,256
    col=0
    for k,u in enumerate(upper):
        center=centers[k]
        hinges=[(0,2,0)]
        for decade in range(10,u+1,10):
            hinges.extend([(decade-1,-20,0),(decade,20,0)])
        for threshold,linear,quadratic in hinges:
            integral=math.floor(threshold)
            W["h0.mlp.w1"][val(k),col]=-64*scale
            W["h0.mlp.w1"][14,col]=-scale*(integral-center)/16
            W["h0.mlp.w2"][col,0:4]=[
                outscale*linear*(33/64),0,
                2*outscale*quadratic,0]
            col+=1
    # A shared neuron supplies the constant quadratic readout coefficient.
    W["h0.mlp.w1"][14,col] = scale/16
    W["h0.mlp.w2"][col,2] = -2*outscale
    col += 1
    assert col==cfg["d_mlp"]

    # Preserve zero residual sum using one shared balancing coordinate.
    # Paired uniform/biased coefficients remain proportional after bf16 rounding.
    positives = [val(k) for k in range(8)]
    negatives = [val(k)+1 for k in range(8)]
    W["h0.attn.wo"][:,90] = -W["h0.attn.wo"][:,positives].sum(axis=1,dtype=np.float64)
    W["h0.attn.wo"][:,negatives] = 0
    W["h0.attn.bo"][90] = -W["h0.attn.bo"][positives].sum(dtype=np.float64)
    W["h0.attn.bo"][negatives] = 0
    assert W["h0.attn.bo"][90] == -123
    return cfg,W

if __name__=="__main__":
    cfg,W=build()
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    for name,array in W.items():
        np.save(out/(name+".npy"),array)
