
import numpy as np
import torch
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state_signal", "code"))  # lgarch.py
import lgarch

def make(LMAX):
    h = int(np.log2(LMAX))
    assert LMAX == 2**h and 32 <= LMAX <= 128
    def prime(p):
        return p >= 2 and all(p%d for d in range(2, int(p**.5)+1))
    primes = []
    k = 1
    while len(primes) < 2:
        p = k*LMAX+1
        if prime(p): primes.append(p)
        k += 1
    p0,p1 = primes
    assert p0*p1 > 81*(LMAX//2)
    def root(p):
        factors = [q for q in range(2,p) if (p-1)%q == 0 and prime(q)]
        return next(g for g in range(2,p) if all(pow(g,(p-1)//q,p)!=1 for q in factors))
    roots = [root(p) for p in primes]
    B = 1
    while 2*(4**B-1)//3 < max(primes)-1:
        B += 1
    C = 2*h+2
    DEC,SUM,SET = C,C+1,C+2
    jumps = list(range(C+3,C+3+h))
    OUT,DONE = C+3+h,C+4+h
    NS,NP = DONE+1,B+2
    POINT,CRT = h,2*h+1
    fwd = list(range(h))
    inv = list(range(h+1,2*h+1))
    R={}
    def alloc(name, n=1):
        if n==1:
            R[name]=len(R)
            return name
        return [alloc(name+str(j)) for j in range(n)]
    r=alloc("r",2); out=alloc("out",10)
    pos=alloc("pos"); pos2=alloc("pos2")
    targets=alloc("target",h)
    tf=[alloc("tf"+str(j)+"_",h) for j in range(2)]
    ti=[alloc("ti"+str(j)+"_",h) for j in range(2)]
    q=alloc("query"); half=alloc("half"); tw=alloc("tw",2)
    st=alloc("stage",NS); ph=alloc("phase",NP)
    x=alloc("x",2); y=alloc("y",2); z=alloc("z",2)
    u=alloc("u",2); bit=alloc("bit",2); active=alloc("active")
    rr=alloc("readr",2); rg=alloc("readG"); rp=alloc("readptr"); rpos=alloc("readpos")
    planes=alloc("plane",4); readplanes=alloc("readplane",3)
    G=alloc("G"); ptr=alloc("ptr")
    D=16*((2*len(R)+2+15)//16)
    assert D<=64+32*h and 2*len(R)+2<=D, (len(R),D)
    nr=len(R)
    def E(name):
        a=np.zeros(nr+1)
        a[R[name]]=1
        return a
    def cn(v):
        a=np.zeros(nr+1); a[-1]=v
        return a
    one=cn(1)
    def total(names): return sum((E(s) for s in names),cn(0))
    def gate(*args): return sum(args,cn(0))-cn(len(args)-1)
    sf,si=total([st[j] for j in fwd]),total([st[j] for j in inv])
    sa=total(st[:C])
    pi,pb,pf=E(ph[0]),total(ph[1:B+1]),E(ph[-1])
    gi=gate(pi,sa); gf=gate(pf,sf); giv=gate(pf,si)
    gfinit,givinit=gate(pi,sf),gate(pi,si)
    gpoint,gcrt=[gate(pi,E(st[j])) for j in (POINT,CRT)]
    gcrtfinal=gate(pf,E(st[CRT]))
    boundcoef=81*(LMAX//2)
    class MLP:
        def __init__(self):
            self.a=[]; self.b=[]
        def relu(self,e,outs):
            v=np.zeros(nr)
            for name,c in outs.items(): v[R[name]]+=c
            if not np.any(v): return
            self.a.append(e.copy()); self.b.append(v)
        def lin(self,e,outs,g=None,bound=4096):
            if g is None:
                self.relu(e,outs); self.relu(-e,{k:-v for k,v in outs.items()})
            else:
                off=(2*bound+4)*(g-cn(.5))
                self.relu(e+off,outs); self.relu(off,{k:-v for k,v in outs.items()})
        def assign(self,name,e,g=None,bound=4096):
            self.lin(e-E(name),{name:1},g,bound)
        def step(self,e,outs,g=None,bound=4096):
            if g is not None: e=e-(bound+4)*(one-g)
            self.relu(e+cn(.25),{k:2*v for k,v in outs.items()})
            self.relu(e-cn(.25),{k:-2*v for k,v in outs.items()})
        def indicator(self,conds,outs):
            self.step(sum(conds,cn(0))-cn(len(conds)-.5),outs)
        def peak(self,e,value,outs,g=None,bound=4096):
            self.step(e-cn(value-.5),outs,g,bound)
            self.step(e-cn(value+.5),{k:-v for k,v in outs.items()},g,bound)
        def lookup(self,e,values,outs,g=None,bound=4096):
            vv=np.asarray(values,dtype=np.float64)
            if vv[0]: self.lin(cn(vv[0]),outs,g,bound=abs(vv[0])+1)
            for j in np.nonzero(np.diff(vv))[0]+1:
                self.step(e-cn(j-.5),{k:v*(vv[j]-vv[j-1]) for k,v in outs.items()},g,bound)
        def masked_read(self,e,outs,stage,threshold):
            off=32*(E(pos)-cn(threshold-.5))-(64*LMAX+64)*(one-stage)
            self.relu(e+off,outs); self.relu(off,{k:-v for k,v in outs.items()})
        def finish(self):
            m=len(self.a)
            aa=np.asarray(self.a); bb=np.asarray(self.b)
            w1=np.zeros((D,m),np.float32); w2=np.zeros((m,D),np.float32)
            w1[0:2*nr:2]=aa[:,:nr].T*512
            w1[1:2*nr:2]=-aa[:,:nr].T*512
            w2[:,0:2*nr:2]=bb/1024
            w2[:,1:2*nr:2]=-bb/1024
            return w1,(aa[:,-1]*1024).astype(np.float32),w2
    ml=[MLP() for _ in range(4)]
    a=ml[0]
    a.assign(half,cn(0),gi,bound=2)
    a.step(E(pos)-E(q),{half:1},gi,bound=2*LMAX+2)
    a.assign(active,cn(0),bound=2)
    a.peak(E(rpos)-E(q),0,{active:1},gate(pi,sf+si),bound=2*LMAX+2)
    a.lin(one,{active:1},gate(pi,total([st[POINT],st[CRT]])),bound=2)
    for j,p in enumerate(primes):
        a.assign(bit[j],cn(0),bound=4)
        a.lin(E(y[j]),{bit[j]:1},pb,bound=2*p)
        a.lin(-E(y[j]),{y[j]:1},pb,bound=2*p)
        a.lookup(E(y[j]),(np.arange(p)+1)//4,{y[j]:1,bit[j]:-4},pb,bound=2*p+2)
        a.assign(u[j],cn(0),pb,bound=2*p+4)
        a.lookup(E(x[j]),(2*np.arange(p))%p,{u[j]:1},pb,bound=2*p+4)
        a.lin(-E(y[j]),{y[j]:1},gi,bound=2*p)
        a.lin(E(tw[j]),{y[j]:1},gate(pi,sf+si),bound=2*p)
        a.lin(E(rr[j]),{y[j]:1},gpoint,bound=2*p)
        a.lin(cn(1 if j==0 else pow(p0,-1,p1)),{y[j]:1},gcrt,bound=p+2)
        a.lin(-E(u[j]),{u[j]:1},giv,bound=2*p+2)
        a.lin(E(r[j])+E(z[j]),{u[j]:1},gate(giv,one-E(half)),bound=2*p+2)
        a.lin(E(rr[j])-E(z[j])+cn(p),{u[j]:1},gate(giv,E(half)),bound=2*p+2)
    a.assign(u[0],cn(0),gcrtfinal,bound=2*p0+2)
    a.lookup(E(r[0]),np.arange(p0),{u[0]:1},gcrtfinal,bound=2*p0+2)
    a.assign(u[1],cn(0),gcrtfinal,bound=2*p1+2)
    a.lookup(E(z[1]),np.arange(boundcoef//p0+1),{u[1]:1},gcrtfinal,bound=2*p1+2)
    for j,base in enumerate((10,100,1000),1):
        a.assign(planes[j],cn(0),E(st[DEC]),bound=boundcoef+2)
        for v in range(base,boundcoef+1,base):
            a.step(E(planes[0])-cn(v-.5),{planes[j]:1},E(st[DEC]),bound=boundcoef+2)
    a.assign(u[0],E(planes[0]),E(st[SUM]),bound=boundcoef+2)
    for j in range(1,4):
        a.masked_read(E(readplanes[j-1]),{u[0]:1},E(st[SUM]),2*j)
    gs=E(st[SET])+E(st[OUT])
    a.assign(u[0],E(x[0]),gs,bound=2*p0+4)
    a.masked_read(E(rg),{u[0]:1},gs,2)
    gjump=total([st[j] for j in jumps])
    a.assign(G,E(rg),gjump,bound=4)
    a.assign(ptr,E(rp),gjump,bound=2*LMAX+4)

    a=ml[1]
    for j,p in enumerate(primes):
        a.assign(u[j],E(z[j]),pb,bound=2*p+4)
        a.lin(E(x[j]),{u[j]:1},gate(pb,E(bit[j])),bound=2*p+4)
        a.lin(E(u[j])-E(x[j]),{u[j]:1},gate(pb,E(bit[j])-one),bound=2*p+4)
        a.lin(cn(p)-E(x[j]),{u[j]:1},gate(pb,-E(bit[j])),bound=2*p+4)
        a.assign(u[j],cn(0),gi,bound=2*p+4)
        a.lin(E(r[j])+E(rr[j]),{u[j]:1},gate(gfinit,one-E(half)),bound=2*p+4)
        a.lin(E(rr[j])-E(r[j])+cn(p),{u[j]:1},gate(gfinit,E(half)),bound=2*p+4)
        a.lin(E(rr[j]),{u[j]:1},gate(givinit,one-E(half)),bound=2*p+4)
        a.lin(E(r[j]),{u[j]:1},gate(givinit,E(half)),bound=2*p+4)
        a.lin(E(r[j]),{u[j]:1},gpoint,bound=2*p+4)
        ce=E(r[0]) if j==0 else E(r[1])-E(r[0])+cn(p1)
        a.lin(ce,{u[j]:1},gcrt,bound=2*p+4)
        # Inverse normalization: qh=floor(raw/2), parity=raw-2*qh.
        a.assign(x[j],cn(0),giv,bound=2*p+4)
        a.assign(bit[j],E(u[j]),giv,bound=2*p+4)
        a.lookup(E(u[j]),np.arange(2*p+1)//2,{x[j]:1,bit[j]:-2},
                 giv,bound=2*p+4)
    for j,base in enumerate((10,100,1000),1):
        a.assign(planes[j],cn(0),E(st[DEC]),bound=boundcoef+2)
        a.lookup(E(planes[j]),np.arange(boundcoef//base+1),{planes[j]:1},
                 E(st[DEC]),bound=boundcoef+2)
    a.assign(x[0],cn(0),E(st[SUM]),bound=2*p0+4)
    a.lookup(E(u[0]),np.arange(37)%10,{x[0]:1},E(st[SUM]),bound=2*p0+4)
    a.assign(G,cn(0),E(st[SUM]),bound=4)
    a.lookup(E(u[0]),np.arange(37)//10,{G:1},E(st[SUM]),bound=2*p0+4)
    a.assign(x[0],cn(0),E(st[SET]),bound=2*p0+4)
    a.lookup(E(u[0]),np.arange(13),{x[0]:1},E(st[SET]),bound=2*p0+4)
    a.assign(G,cn(0),E(st[SET]),bound=4)
    a.step(E(u[0])-cn(9.5),{G:1},E(st[SET]),bound=2*p0+4)
    a.assign(ptr,E(pos),E(st[SET]),bound=2*LMAX+4)
    a.peak(E(u[0]),9,{ptr:-2},E(st[SET]),bound=2*p0+4)
    for digit in range(10):
        a.assign(out[digit],cn(0),E(st[OUT]),bound=4)
        for value in (digit,digit+10):
            if value<=13:
                a.peak(E(u[0]),value,{out[digit]:1},E(st[OUT]),bound=2*p0+4)

    a=ml[2]
    for j,p in enumerate(primes):
        gr=pb+E(active)
        a.assign(u[j],cn(0),gr,bound=2*p+4)
        a.lookup(E(u[j]),np.arange(2*p+1)%p,{u[j]:1},gr,bound=2*p+4)
        a.assign(u[j],E(x[j]),giv,bound=2*p+4)
        a.step(E(bit[j])-cn(.5),{u[j]:(p+1)//2},giv,bound=2*p+4)
    for j in range(3):
        a.assign(planes[j],E(planes[j])-10*E(planes[j+1]),E(st[DEC]),bound=boundcoef+4)

    a=ml[3]
    for j,p in enumerate(primes):
        a.assign(x[j],E(u[j]),gi,bound=2*p+4)
        a.assign(z[j],cn(0),gi,bound=2*p+4)
        a.assign(x[j],cn(0),pb,bound=2*p+4)
        a.lookup(E(x[j]),(4*np.arange(p))%p,{x[j]:1},pb,bound=2*p+4)
        a.assign(z[j],E(u[j]),pb,bound=2*p+4)
        a.assign(r[j],E(z[j]),gate(pf,sf+E(st[POINT])),bound=2*p+4)
        a.assign(r[j],E(u[j]),giv,bound=2*p+4)
        a.step(E(u[j])-cn(p-.5),{r[j]:-p},giv,bound=2*p+4)
    a.assign(planes[0],cn(0),gcrtfinal,bound=boundcoef+4)
    a.lin(E(u[0]),{planes[0]:1},gcrtfinal,bound=2*p0+4)
    a.lin(E(u[1]),{planes[0]:p0},gcrtfinal,bound=2*p1+4)
    for j in range(4):
        a.assign(planes[j],cn(0),E(st[DEC]),bound=boundcoef+4)
        a.lookup(E(planes[j]),np.arange(10),{planes[j]:1},E(st[DEC]),bound=boundcoef+4)
    # The phase wheel has I, B radix passes, F; missing butterflies jump directly to I.
    for j in range(NP):
        a.assign(ph[j],cn(0),bound=3)
        if j==0:
            a.indicator([pf,sa],{ph[j]:1})
            a.indicator([pi,one-E(active)],{ph[j]:1})
        elif j==1:
            a.indicator([pi,E(active),sa],{ph[j]:1})
        else:
            fast=E(st[h-1])+E(st[h+1])
            if j==2:
                a.indicator([E(ph[j-1]),sa,one-fast],{ph[j]:1})
            else:
                a.indicator([E(ph[j-1]),sa],{ph[j]:1})
            if j==B+1:
                a.indicator([E(ph[1]),fast],{ph[j]:1})
    # Advance stage on F or a missing-partner initialization; postprocessing advances each pass.
    for j in range(C):
        for cond in ([E(st[j]),pf],[E(st[j]),pi,one-E(active)]):
            a.indicator(cond,{st[j]:-1,st[j+1]:1})
    for j in range(C,DONE):
        a.lin(E(st[j]),{st[j]:-1,st[j+1]:1},bound=2)
    def stage_constants(j):
        if j<h:
            s=h-1-j
            return E(targets[s]),[E(tf[t][s]) for t in range(2)]
        if h<j<=2*h:
            s=j-h-1
            return E(targets[s]),[E(ti[t][s]) for t in range(2)]
        if j==POINT: return E(ptr),[cn(0),cn(0)]
        return E(pos),[cn(0),cn(0)]
    for old in range(C):
        qe,twe=stage_constants(old+1)
        for gg in (gate(E(st[old]),pf),gate(E(st[old]),pi,one-E(active))):
            a.assign(q,qe,gg,bound=4*LMAX+4)
            for j,p in enumerate(primes):
                a.assign(tw[j],twe[j],gg,bound=2*p+4)
    # Main read addresses for carries.
    a.assign(q,E(pos)-cn(2),E(st[SUM]),bound=4*LMAX+4)
    a.assign(q,E(ptr),E(st[SET]),bound=4*LMAX+4)
    for j in jumps:
        qe=E(pos)-cn(2) if j==jumps[-1] else E(ptr)
        a.assign(q,qe,E(st[j]),bound=4*LMAX+4)
    for name in rr+[rg,rp,rpos]+readplanes:
        a.assign(name,cn(0),bound=boundcoef+4)

    parts=[m.finish() for m in ml]
    M=max(v[0].shape[1] for v in parts)
    assert M<=4096, (M,[v[0].shape[1] for v in parts])
    cfg=dict(n_layer=4,n_head=4,d_head=8,d_model=D,d_mlp=M,n_pos=2*LMAX,scratch=[0,0,0])
    W={k:np.zeros(sh,np.float32) for k,sh in lgarch.shapes(cfg).items()}
    anchor=2.**30
    gamma=anchor*np.sqrt(2/D)
    for l in range(4):
        for name in ("ln1","ln2"): W[f"h{l}.{name}.w"][:]=gamma
        w1,b1,w2=parts[l]; m=w1.shape[1]
        W[f"h{l}.mlp.w1"][:,:m]=w1
        W[f"h{l}.mlp.b1"][:m]=b1
        W[f"h{l}.mlp.w2"][:m]=w2
    def put(arr,row,name,value):
        k=2*R[name]; arr[row,k]=value; arr[row,k+1]=-value
    for digit in range(10):
        for name in r: put(W["wte"],digit,name,digit)
        put(W["wte"],digit,out[digit],1)
    for name in out:
        W["lnf.w"][2*R[name]:2*R[name]+2]=gamma
    for t in range(2*LMAX):
        i,lane=divmod(t,2)
        W["wpe"][t,-2:]=[anchor,-anchor]
        put(W["wpe"],t,pos,t); put(W["wpe"],t,pos2,t*t)
        put(W["wpe"],t,ptr,t^1)
        put(W["wpe"],t,st[0],1); put(W["wpe"],t,ph[0],1)
        for s in range(h):
            put(W["wpe"],t,targets[s],2*(i^(1<<s))+lane)
            for j,(p,rootp) in enumerate(zip(primes,roots)):
                rootspan=pow(rootp,(p-1)//(1<<(s+1)),p)
                exponent=i% (1<<s)
                fw=pow(rootspan,exponent,p) if i & (1<<s) else 1
                iv=pow(pow(rootspan,-1,p),exponent,p)
                put(W["wpe"],t,tf[j][s],fw)
                put(W["wpe"],t,ti[j][s],iv)
                if s==h-1: put(W["wpe"],t,tw[j],fw)
        put(W["wpe"],t,q,2*(i^(1<<(h-1)))+lane)
    H,DH=4,8
    A=H*DH
    qkv=W["h0.attn.wqkv"]; bqkv=W["h0.attn.bqkv"]; wo=W["h0.attn.wo"]
    def col(name,col,value):
        k=2*R[name]; qkv[k,col]+=value/2; qkv[k+1,col]-=value/2
    def output(colno,name,value=1):
        k=2*R[name]; wo[colno,k]+=value; wo[colno,k+1]-=value
    sharp=32.
    for head in range(H):
        base=head*DH
        col(q if head==0 else pos,base,2*sharp*np.sqrt(DH))
        if head: bqkv[base]=-4*head*sharp*np.sqrt(DH)
        bqkv[base+1]=-sharp*np.sqrt(DH)
        col(pos,A+base,1); col(pos2,A+base+1,1)
        if head==0:
            for j,(src,dst) in enumerate(zip(r+[G,ptr,pos],rr+[rg,rp,rpos])):
                col(src,2*A+base+j,1); output(base+j,dst)
        else:
            col(planes[head],2*A+base,1); output(base,readplanes[head-1])
    meta=dict(R=R,primes=primes,roots=roots,bits=B,stages=NS,phases=NP,
              stage_names=dict(POINT=POINT,CRT=CRT,DEC=DEC,SUM=SUM,SET=SET,OUT=OUT,DONE=DONE),
              mlp_units=[v[0].shape[1] for v in parts],registers=nr)
    return W,cfg,meta

if __name__=="__main__":
    outdir= torch.sys.argv[1]
    lmax=int(torch.sys.argv[2])
    torch.os.makedirs(outdir,exist_ok=True)
    W,cfg,meta=make(lmax)
    lgarch.save(torch.os.path.join(outdir,"weights.npz"),W,cfg)
    print(cfg)
    print(meta)
