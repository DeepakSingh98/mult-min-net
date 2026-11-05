"""Hand-set 364-nonzero, strict-margin 42-pass multiplication transformer.
Only arch.py supplies model definitions. No training or weight files.
build_paired specifies a logical circuit; compress and pack are algebraic
representation changes, not weight search or empirical pruning.
"""

# Core adapted from the supplied earlier-session build; reductions are hand-derived.
import sys, json, importlib.util
from pathlib import Path
import numpy as np

CORE_CFG = dict(n_layer=1, loop=42, n_head=2, d_head=8, d_model=50, d_mlp=48)
CFG = dict(CORE_CFG, d_head=7, d_mlp=44)

def build_paired():
    spec=importlib.util.spec_from_file_location("task_arch", Path(__file__).resolve().parents[1] / "arch.py")
    arch=importlib.util.module_from_spec(spec); spec.loader.exec_module(arch)
    W={n:np.zeros(s,np.float32) for n,s in arch.shapes(dict(CORE_CFG,d_model=128)).items()}
    cursor=0
    def alloc(n=1):
        nonlocal cursor
        a=list(range(cursor,cursor+2*n,2)); cursor+=2*n
        return a if n!=1 else a[0]
    digit,triangle=alloc(),alloc()
    pos={p:alloc() for p in list(range(5,9))+list(range(11,19))}
    keyj,keyj2,valid=alloc(),alloc(),alloc()
    anchor=alloc()
    off=alloc(2)
    counter=alloc()
    ret=alloc(2)
    products=alloc(4)
    prod,sums=alloc(7),alloc(7)
    ans,asum=[],[]
    total,one=alloc(),alloc()
    assert cursor<=128,cursor
    def pair(row,idx,val):
        row[idx]+=val; row[idx+1]-=val
    def read(col,idx,val):
        pair(col,idx,val/2)
    for d in range(10):
        pair(W["wte"][d],digit,d); pair(W["wte"][d],triangle,d*(d-1)/2)
    for p in range(20):
        row=W["wpe"][p]
        if p in pos: pair(row,pos[p],1)
        if p<4:
            pair(row,keyj,p); pair(row,keyj2,p*p)
            pair(row,valid,1)

    for token in range(14):
        pair(W["wte"][token],anchor,2**20)
    pair(W["h0.ln2.b"],counter,-41)
    def ln(name,indices):
        for idx in set(indices): W[name][idx:idx+2]=2**17
    ln("h0.ln1.w",[digit,keyj,keyj2,valid,counter]+prod+ans)
    ln("h0.ln2.w",[digit,triangle]+list(pos.values())+off+ret[:1]+products+
       sums+asum+[total,counter])
    for j,idx in enumerate(off):
        pair(W["h0.ln2.b"],idx,2*([4,5][j]))
    W["lnf.w"][digit:digit+2]=2**20
    W["lnf.w"][triangle:triangle+2]=2**20
    Q,O=W["h0.attn.wqkv"],W["h0.attn.wo"]
    A=16
    # Address a_(3-counter) by a quadratic position score.
    W["h0.attn.bqkv"][0]=192
    read(Q[:,0],counter,-64)
    W["h0.attn.bqkv"][1]=-32
    W["h0.attn.bqkv"][2]=32
    read(Q[:,A],keyj,32)
    read(Q[:,A+1],keyj2,32)
    read(Q[:,A+2],valid,1024)
    for j,idx in enumerate([digit]):
        read(Q[:,2*A+j],idx,1); pair(O[j],ret[j],1)
    for j,idx in enumerate(prod):
        read(Q[:,2*A+8+j],idx,1)
        pair(O[8+j],sums[j],222/256)
    u=0
    def neuron(coeff,bias,outputs,scale=32):
        nonlocal u
        for idx,val in coeff.items(): read(W["h0.mlp.w1"][:,u],idx,scale*val)
        W["h0.mlp.b1"][u]=scale*bias
        for idx,val in outputs.items(): pair(W["h0.mlp.w2"][u],idx,val/scale)
        u+=1
    # Analytic small-argument GELU multiplication. The rounded scale
    # is compensated by routing factors 37/32 and 222/256. No fitting.
    import torch, math
    s=2.**-10
    exact=1/(math.sqrt(2/math.pi)*s*s)
    hi=float(torch.tensor(exact).bfloat16().float())
    for factor in [hi]:
        neuron({ret[0]:s,digit:s},0,{products[0]:factor},scale=1)
        neuron({ret[0]:s},0,{products[0]:-factor},scale=1)
        neuron({digit:s},0,{products[0]:-factor},scale=1)
    neuron({ret[0]:1},0,{ret[0]:-1})
    # Shift product history: after four passes it is (a0*b,a1*b,a2*b,a3*b).
    for j in range(4):
        out={products[j]:-1}
        if j<3: out[products[j+1]]=1
        neuron({products[j]:1},0,out)
    # Pass 5: route by the b-digit's position.
    for j in range(4):
        for i in range(4):
            neuron({products[i]:1,pos[5+j]:256,off[0]:-256},-256,{prod[i+j]:37/32})
    for r in range(7):
        neuron({digit:1,pos[12+r]:256,off[0]:-256},-256,{prod[r]:-37/32})
    # Pass 6: obtain the column integer, to within a small fraction.
    for k in range(8):
        coeff={pos[11+k]:64,off[1]:-64}
        for r in range(max(0,k-2),min(k,6)+1): coeff[sums[r]]=10.**(r-k)
        neuron(coeff,-64,{total:12+k},scale=128)
    # Passes 7..41: subtract ten once whenever total >= 10.
    # Earlier passes have total zero, and later exhausted values are < 10.
    for offset,sign in [(.75,-40),(.5,40)]:
        neuron({total:1},-10+offset,{total:sign})
    # Pass 42: tied quadratic readout.
    readout_start=u
    neuron({total:1,counter:255},1.625,{digit:1})
    neuron({digit:1,counter:255},1,{digit:-1})
    neuron({triangle:1,counter:255},0,{triangle:-1})
    # Stored timers are 2*(|counter-target|-target); LN bias adds 2*target.
    # Counter has LN2 bias -41, so hinge biases compensate for that shift.
    for j,target in enumerate([4,5]):
        for offset,sign in [(.75,8),(.25,-8)]:
            neuron({counter:1},41-target+offset,{off[j]:sign})
        pair(W["h0.mlp.b2"],off[j],-2)
    pair(W["h0.mlp.b2"],counter,1)
    assert u==CORE_CFG["d_mlp"],u
    return W,dict(digit=digit,triangle=triangle,pos=pos,off=off,counter=counter,
                  ret=ret,products=products,prod=prod,ans=ans,sums=sums,
                  asum=asum,total=total,one=one,used=u,readout_start=readout_start)

def compress(W, layout):
    import torch
    # All delivered weights are themselves bfloat16 numbers.
    def bf(x):
        return torch.tensor(x,dtype=torch.float32).bfloat16().float().numpy()
    W={n:bf(v) for n,v in W.items()}
    anchor=34  # allocation: 2 token pairs, 12 position pairs, 3 key pairs
    assert np.max(np.abs(W["wte"][:,anchor]))==2**20
    keep_odd={1,3,anchor+1}
    refs=[5,7,9,11]
    # Residual writes and embeddings: discard internal negative copies.
    for name in ["wte","wpe","h0.attn.wo","h0.mlp.w2"]:
        for idx in range(1,128,2):
            if idx not in keep_odd: W[name][:,idx]=0
    for name in ["h0.attn.bo","h0.mlp.b2"]:
        for idx in range(1,128,2):
            if idx not in keep_odd: W[name][idx]=0
    for which,projection in [("ln1","attn.wqkv"),("ln2","mlp.w1")]:
        g=W["h0."+which+".w"]; b=W["h0."+which+".b"]
        # Input projections read positive coordinates only.
        g[1::2]=0; b[1::2]=0
        mat=W["h0."+projection]
        mat[0::2] *= 2
        mat[1::2] = 0
        variable_rows=g != 0
        for col in range(mat.shape[1]):
            remainder=-float(mat[variable_rows,col].astype(np.float64).sum())
            for ref in refs:
                if remainder==0: break
                part=float(bf(remainder))
                mat[ref,col]=part
                remainder-=part
                g[ref]=2**17
            assert remainder==0, (projection,col,remainder)
    # Input token negative coordinates are still needed by the final tied readout.
    for name,value in W.items():
        assert np.array_equal(value,bf(value)),name
    layout["references"]=refs
    layout["anchor"]=anchor
    return W,layout

def pack(W,layout):
    # Three-coordinate zero-sum token code: d, -d-T(d), T(d).
    W["wte"][:,1]=-W["wte"][:,0]-W["wte"][:,2]
    W["wte"][:,3]=0
    W["lnf.w"][3]=0
    a=layout["readout_start"]
    W["h0.mlp.w2"][a,1]=0
    W["h0.mlp.w2"][a+2,1]=W["h0.mlp.w2"][a+2,3]
    W["h0.mlp.w2"][a+2,3]=0
    # D=2*5^2; anchors +/-5*2^22 imply standard deviation 2^22.
    anchor=layout["anchor"]
    W["wte"][:,anchor:anchor+2] *= 20
    W["h0.ln1.w"] *= 32
    W["h0.ln2.w"] *= 32
    W["lnf.w"] *= 16
    axes={}
    for name,arr in W.items():
        if name in ["wte","wpe","h0.attn.wo","h0.mlp.w2"]:
            axes[name]=1
        elif name in ["h0.attn.wqkv","h0.mlp.w1"]:
            axes[name]=0
        elif arr.ndim==1 and arr.shape[0]==128:
            axes[name]=0
    used=set()
    for name,axis in axes.items():
        arr=W[name]
        if arr.ndim==1: active=np.flatnonzero(arr)
        else: active=np.flatnonzero(np.any(arr!=0,axis=1-axis))
        used.update(map(int,active))
    used=sorted(used)
    assert len(used)<=CORE_CFG["d_model"],len(used)
    mapping={old:new for new,old in enumerate(used)}
    for name,axis in axes.items():
        arr=np.take(W[name],used,axis=axis)
        padding=[(0,0)]*arr.ndim
        padding[axis]=(0,CORE_CFG["d_model"]-len(used))
        W[name]=np.pad(arr,padding).astype(np.float32)
    for key in ["digit","triangle","counter","total","one","anchor"]:
        layout[key]=mapping.get(layout[key])
    layout["pos"]={p:mapping[idx] for p,idx in layout["pos"].items()}
    for key in ["off","ret","products","prod","ans","sums","asum","references"]:
        layout[key]=[mapping[idx] for idx in layout[key] if idx in mapping]
    layout["token_balance"]=mapping[1]
    layout["active_coordinates"]=len(used)
    layout["coordinate_map"]=mapping
    return W,layout

def build():
    W,layout=build_paired()
    W,layout=compress(W,layout)
    W,layout=pack(W,layout)
    # Constant triangular readout: retain only the difference of digit axes.
    d,tri,bal=layout["digit"],layout["triangle"],layout["token_balance"]
    u=layout["readout_start"]+2
    W["wte"][:,bal]=-W["wte"][:,d]
    W["lnf.w"][tri]=0
    W["lnf.b"][tri]=-4
    W["h0.ln2.w"][tri]=0
    W["h0.mlp.w1"][:,u]=0
    W["h0.mlp.b1"][u]=0
    W["h0.mlp.w2"][u,:]=0
    # Balanced pass-5 gates: signal+128*position-129*timer-128.
    for col in range(8,31):
        p=5+(col-8)//4 if col<24 else 12+col-24
        W["h0.mlp.w1"][layout["pos"][p],col]=128*32
        W["h0.mlp.w1"][layout["off"][0],col]=-129*32
        W["h0.mlp.w1"][layout["references"],col]=0
        W["h0.mlp.b1"][col]=-128*32
    # Fold the source-validity bonus into its quadratic address key.
    j2,valid=layout["coordinate_map"][30],layout["coordinate_map"][32]
    W["wpe"][:,j2]-=32*W["wpe"][:,valid]
    W["wpe"][:,valid]=0
    W["h0.ln1.w"][valid]=0
    W["h0.attn.wqkv"][:,18]=0
    W["h0.attn.bqkv"][2]=0
    # Strict separation of marker logits, using existing anchor entries.
    anchor=layout["anchor"]
    W["wte"][10:,anchor:anchor+2]*=-1
    W["lnf.b"][anchor]=2.**-24
    # Factor position-minus-one across every position-gated unit.
    for idx in layout["pos"].values():
        W["h0.ln2.b"][idx]=-1
    W["h0.mlp.b1"][8:39]=0

    # Squared clocks: at pass index r, expose (r-4)^2 and (r-5)^2.
    off0,off1=layout["off"]
    ctr=layout["counter"]
    W["h0.ln2.b"][off0]=16
    W["h0.ln2.b"][off1]=25
    W["h0.mlp.w1"][:,44:48]=0
    W["h0.mlp.b1"][44:48]=0
    W["h0.mlp.w2"][44:48,:]=0
    W["h0.mlp.w1"][ctr,44]=32
    W["h0.mlp.w1"][layout["references"][0],44]=-32
    W["h0.mlp.b1"][44]=32*41
    W["h0.mlp.w2"][44,off0]=2/32
    W["h0.mlp.w2"][44,off1]=2/32
    W["h0.mlp.b2"][off0]=-7
    W["h0.mlp.b2"][off1]=-9
    # Difference-form analytic GELU multiplication.
    W["h0.mlp.w1"][d,0]*=-1
    W["h0.mlp.w1"][layout["references"],0]=0
    W["h0.mlp.w1"][:,2]*=-1
    W["h0.mlp.w2"][0:3,:]*=-1
    # Coarse carry coefficients with one bf16-exact mean reference each.
    for k in range(8):
        col=31+k
        for r in range(max(0,k-2),min(k,6)+1):
            delta=k-r
            coefficient=1 if delta==0 else (
                (3/32 if k==1 else 51/512) if delta==1 else 5/512)
            W["h0.mlp.w1"][layout["sums"][r],col]=128*coefficient
        W["h0.mlp.w1"][layout["references"],col]=0
        W["h0.mlp.w1"][layout["references"][0],col]=-W["h0.mlp.w1"][:,col].sum()
    ref2=layout["references"][1]
    assert not np.any(W["h0.mlp.w1"][ref2])
    W["h0.ln2.w"][ref2]=0
    W["h0.mlp.b1"][39]=32*(-9.125)
    W["h0.mlp.b1"][40]=32*(-9.375)
    W["h0.mlp.b1"][41]=32*1.75
    # Sharper modulo hinges prevent repeated GELU-tail drift.
    W["h0.mlp.w1"][:,39:41]*=2
    W["h0.mlp.b1"][39:41]*=2
    W["h0.mlp.w2"][39:41,:]*=0.5

    # Delete identically unused slots, not empirically selected parameters.
    qkv_keep=[part*16+head*8+j for part in range(3)
              for head in range(2) for j in range(7)]
    head_keep=[head*8+j for head in range(2) for j in range(7)]
    units=list(range(43))+[44]
    W["h0.attn.wqkv"]=W["h0.attn.wqkv"][:,qkv_keep].copy()
    W["h0.attn.bqkv"]=W["h0.attn.bqkv"][qkv_keep].copy()
    W["h0.attn.wo"]=W["h0.attn.wo"][head_keep,:].copy()
    W["h0.mlp.w1"]=W["h0.mlp.w1"][:,units].copy()
    W["h0.mlp.b1"]=W["h0.mlp.b1"][units].copy()
    W["h0.mlp.w2"]=W["h0.mlp.w2"][units,:].copy()
    layout["used"]=44
    layout["timer_unit"]=43
    layout["references"]=[layout["references"][0]]
    layout["active_coordinates"]=43
    return W,layout

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    W,_=build();(out/"config.json").write_text(json.dumps(CFG,indent=2))
    for name,value in W.items():np.save(out/(name+".npy"),value)
if __name__=="__main__":main()
