
"""487-nonzero strict-margin multiplication transformer.
Hand-set constants only: no training, fitting, or weight search.
Adapted from the supplied 511-nonzero two-block construction.
"""
import sys, math, json, importlib.util
from pathlib import Path
import numpy as np

def build():
    spec=importlib.util.spec_from_file_location("task_arch", Path(__file__).resolve().parents[1] / "arch.py")
    arch=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    cfg=dict(n_layer=2,loop=1,n_head=8,d_head=2,d_model=512,d_mlp=69)
    W={name:np.zeros(shape,dtype=np.float32) for name,shape in arch.shapes(cfg).items()}
    A,DH=16,2
    digit,square,anchor,position,position2=0,2,4,6,7
    operands=list(range(8,16))
    queries=list(range(16,20))
    column,balance=20,23
    R=2.**18
    std=R/16  # sqrt(2 R^2 / 512)

    def pair(row,idx,value):
        row[idx]+=value
        row[idx+1]-=value

    def bf(value):
        """IEEE float32 -> bfloat16 -> float32, round-to-nearest ties-to-even."""
        bits=np.asarray(np.float32(value)).view(np.uint32).item()
        bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
        return np.asarray(np.uint32(bits)).view(np.float32).item()

    # Balanced digit and quadratic readout codes.
    for d in range(10):
        pair(W["wte"][d],digit,d)
        W["wte"][d,square]=d*(d-1)
        W["wte"][d,balance]=-d*(d-1)
    # Only digit embeddings contain anchors, enabling a shared +1 digit logit.
    for token in range(10):
        pair(W["wte"][token],anchor,R)
    # Fixed-marker positions restore identical input residuals.
    for p in (9,10,11):
        pair(W["wpe"][p],anchor,R)

    # Position 19 is causally irrelevant to all eight scored predictions.
    for p in range(19):
        W["wpe"][p,position]=p-4
        W["wpe"][p,position2]=(p-4)**2
        for i in range(4):
            j=p-11-i
            if 0<=j<4:
                W["wpe"][p,queries[i]]=j+1
    W["wpe"][:,balance]=-W["wpe"].sum(axis=1)

    def ln(name,coords):
        for idx in coords:
            W[name+".w"][idx]=std
    ln("h0.ln1",[digit,position,position2]+queries)
    ln("h0.ln2",operands)
    ln("h1.ln1",[digit,column,position,position2])
    ln("h1.ln2",[digit,digit+1])
    W["h1.ln2.b"][digit]=15/16
    W["h1.ln2.b"][digit+1]=-7/16
    W["lnf.w"][digit:digit+2]=std
    W["lnf.b"][digit]=-5/8
    W["lnf.b"][square]=-1
    W["lnf.b"][anchor]=1/R

    # Layer 0 attention: four fixed a_i reads, four shifted b_{k-i} reads.
    # Scores are (64/sqrt(2))*(2*t*z-z*z), peaking at source z=t.
    Q=W["h0.attn.wqkv"]
    B=W["h0.attn.bqkv"]
    for h in range(8):
        base=h*DH
        if h<4:
            B[base]=128*(h-4)
        else:
            Q[queries[h-4],base]=128
        B[base+1]=-64
        Q[position,A+base]=1
        Q[position2,A+base+1]=1
        Q[digit,2*A+base]=1
        W["h0.attn.wo"][base,operands[h]]=1
        W["h0.attn.wo"][base,balance]=-1

    # Layer 0 MLP: sum_i GELU(e*(a_i+b_i))-GELU(e*(a_i-b_i)),
    # minus e*sum_i b_i using one saturated GELU on the nonnegative sum.
    # Leading product scale is 4*c*e^2*C = sqrt(8/pi), c=1/sqrt(2*pi).
    e,C=2.**-10,2.**20
    u=0
    for i in range(4):
        for sb,sign in [(1,1),(-1,-1)]:
            W["h0.mlp.w1"][operands[i],u]=e
            W["h0.mlp.w1"][operands[4+i],u]=e*sb
            W["h0.mlp.w2"][u,column]=C*sign
            W["h0.mlp.w2"][u,balance]=-C*sign
            u+=1
    # The b-digit sum is nonnegative. GELU(16*s)=16*s for integer s>=0,
    # up to negligible soft-selection and normalization error near zero.
    for i in range(4):
        W["h0.mlp.w1"][operands[4+i],u]=16
    W["h0.mlp.w2"][u,column]=-C*e/16
    W["h0.mlp.w2"][u,balance]=C*e/16
    u+=1
    assert u==9

    # Layer 1 attention: T = S_k + .1*(S_{k-1}-c_{k-1})
    #                              + .01*(S_{k-2}-c_{k-2}).
    Q=W["h1.attn.wqkv"]
    B=W["h1.attn.bqkv"]
    for h in range(3):
        base=h*DH
        Q[position,base]=128
        B[base]=-128*h
        B[base+1]=-64
        Q[position,A+base]=1
        Q[position2,A+base+1]=1
        factor=math.sqrt(math.pi/8)*10.**(-h)
        # One value channel: factor represented as a product of two BF16 values.
        # Round factor upward on its BF16 binade, then round the quotient.
        step=2.**(math.floor(math.log2(factor))-7)
        outer=math.ceil(factor/step)*step
        inner=bf(factor/outer)
        Q[column,2*A+base]=inner
        pair(W["h1.attn.wo"][base],digit,outer)
        if h<2:
            Q[digit,2*A+base+1]=1
            pair(W["h1.attn.wo"][base+1],digit,
                 -1.1 if h==0 else -0.01)

    # The residual readout pair is (T,-T). h1.ln2.b supplies the two
    # hinge offsets; the final readout centers omitted carry by +3/16.

    # Layer 1 MLP: 34 paired decade resets, followed by one isolated-top hinge.
    # Each paired reset subtracts 20 from the readout difference.
    # The hinges have half-unit separation; K=64 sharpens them.
    K=64.
    u=0
    for m in range(1,35):
        for first,sign in [(True,-1),(False,1)]:
            W["h1.mlp.w1"][digit if first else digit+1,u]=K if first else -K
            W["h1.mlp.b1"][u]=-K*10*m
            W["h1.mlp.w2"][u,digit]=sign*40/K
            u+=1
    # Only 9999*9999 attains N=350; every other column N is <=341.
    # This isolated case needs only a push into digit zero's half-line.
    W["h1.mlp.w1"][digit,u]=K
    W["h1.mlp.b1"][u]=-K*344
    W["h1.mlp.w2"][u,digit]=-4/K
    u+=1
    assert u==69
    assert sum(np.count_nonzero(v) for v in W.values())==487
    return cfg,W

def main():
    cfg,W=build()
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    for name,value in W.items():
        np.save(out/(name+".npy"),value)
if __name__=="__main__":
    main()
