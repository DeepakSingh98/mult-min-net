"""WSP-STATE-SIGNAL Experiment 1: how much does the answer depend on the state at pass t? No training.

  e1.py CASES.json OUT_DIR --inputs DIR --procs N --stop-seconds S

A case is one network copy, one length, one part: `back` (backward sensitivity, P(d), per-pass weight gradient), `sigma`
(top singular value of the state-to-answer Jacobian), `cf` (counterfactual interpolation), `iso` and `grad` (finite perturbation
along noise and along the gradient of M), `reg` (named registers). Each case writes one JSON when it ends; finished cases are
skipped, and no new case starts after --stop-seconds. One torch thread per process. See PROTOCOL.md.

All three network families are autonomous after the embedding: x_{t+1} = F(x_t). `anylen` and `ngpu` are cellular automata.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch_anylen as A1  # noqa: E402
import lgarch as LG  # noqa: E402
import train_gd as TG  # noqa: E402

SHA = {"anylen.npz": "c9fdfea3a2c35c500297c71415da45b0a8cc430143163317ee49acd9222df3dd",
       "ntt.npz": "4a88144218b8b25906520b835c43a34e59ae6f1a72feaae259da2e9c7fe0c3cb"}
REG_ANYLEN = ["A+", "A-", "ref", "one", "DIG", "isStar", "isEOS", "SQ", "Ans", "Ans_in", "Cap", "S", "S_in", "Tl", "Tl_in", "Lsep", "Lsep_in",
              "Lsep_prev", "G", "G_in", "G_prev", "Fb", "Zu", "Zu_in", "Zc", "Zc_in", "ZH", "ZH_in", "Yu", "Yc", "YH", "P", "Fz", "Fz_in",
              "Fz_prev", "Edge"]


def reg_ntt():
    """order of alloc() in the frozen build at LMAX 128 (h = 7, B = 5): logical register -> index, physical coordinates 2i, 2i + 1."""
    h, NS, NP = 7, 28, 7
    names = ["r0", "r1"] + [f"out{j}" for j in range(10)] + ["pos", "pos2"] + [f"target{j}" for j in range(h)]
    names += [f"tf{a}_{j}" for a in range(2) for j in range(h)] + [f"ti{a}_{j}" for a in range(2) for j in range(h)]
    names += ["query", "half", "tw0", "tw1"] + [f"stage{j}" for j in range(NS)] + [f"phase{j}" for j in range(NP)]
    names += ["x0", "x1", "y0", "y1", "z0", "z1", "u0", "u1", "bit0", "bit1", "active", "readr0", "readr1", "readG", "readptr", "readpos"]
    names += [f"plane{j}" for j in range(4)] + [f"readplane{j}" for j in range(3)] + ["G", "ptr"]
    assert len(names) == 113
    return {nm: i for i, nm in enumerate(names)}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rand_like(W, seed):
    """gapb.load_net's random control: GPT-2 init, sd 0.02; LN weight 1 and bias 0; biases zero."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sorted(W.items()):
        if ".ln" in k or k.startswith("lnf"):
            out[k] = torch.ones_like(v) if k.endswith(".w") else torch.zeros_like(v)
        elif k.endswith(("b1", "b2", "bo", "bqkv")):
            out[k] = torch.zeros_like(v)
        else:
            out[k] = 0.02 * torch.randn(v.shape, generator=g, dtype=v.dtype)
    return out


class Anylen:
    """the any-length multiplier, a 1-D cellular automaton on transformer blocks."""
    written = True

    def __init__(self, inputs, n, seed=None, dtype=torch.float32):
        p = os.path.join(inputs, "anylen.npz")
        assert sha(p) == SHA["anylen.npz"]
        self.cfg, W = A1.load(p)
        self.W = {k: v.to(dtype) for k, v in (rand_like(W, seed) if seed is not None else W).items()}
        self.written = seed is None
        self.n, self.K, self.dtype = n, 5 * n + 2, dtype
        self.reg = {nm: [i] for i, nm in enumerate(REG_ANYLEN)}

    def problems(self, N, rng):
        n = self.n
        a, b = rng.integers(0, 10, (N, n)), rng.integers(0, 10, (N, n))
        if n > 1:
            a[:, n - 1], b[:, n - 1] = rng.integers(1, 10, N), rng.integers(1, 10, N)
        a2 = a.copy()
        for r in range(N):
            i = rng.integers(n)
            a2[r, i] = rng.choice([d for d in range(1 if i == n - 1 and n > 1 else 0, 10) if d != a[r, i]])
        t, y = A1.encode(a, b)
        t2, y2 = A1.encode(a2, b)
        t2[:, 2 * n + 3:] = t[:, 2 * n + 3:]         # keep the first problem's teacher-forced answer tokens: Delta is input information only
        T = t.shape[1]
        pos = torch.arange(T)
        dist = (pos[:, None] - pos[None, :]).to(self.dtype)
        self.bias = [(-torch.tensor(self.cfg["alibi"][l], dtype=self.dtype)[:, None, None] * dist).masked_fill(dist < 0, float("-inf"))
                     for l in range(self.cfg["n_layer"])]
        self.p0 = 2 * n + 2
        return dict(t=t, t2=t2, y=torch.as_tensor(y), y2=torch.as_tensor(y2))

    def x0(self, t):
        return self.W["wte"][t]

    def params(self):
        return list(self.W.values())

    def step(self, x):
        W, cfg = self.W, self.cfg
        N, T, _ = x.shape
        D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
        A = H * DH
        for l in range(cfg["n_layer"]):
            h = F.layer_norm(x, (D,), W[f"h{l}.ln1.w"], W[f"h{l}.ln1.b"], A1.EPS)
            qkv = h @ W[f"h{l}.attn.wqkv"] + W[f"h{l}.attn.bqkv"]
            q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
            s = (q @ k.transpose(-1, -2)) / math.sqrt(DH) + self.bias[l]
            z = s.softmax(-1) @ v
            x = x + z.transpose(1, 2).reshape(N, T, A) @ W[f"h{l}.attn.wo"] + W[f"h{l}.attn.bo"]
            h = F.layer_norm(x, (D,), W[f"h{l}.ln2.w"], W[f"h{l}.ln2.b"], A1.EPS)
            x = x + A1.gelu_new(h @ W[f"h{l}.mlp.w1"] + W[f"h{l}.mlp.b1"]) @ W[f"h{l}.mlp.w2"] + W[f"h{l}.mlp.b2"]
        return x

    def read(self, x):
        n = self.n
        return A1.readout(self.W, x)[:, self.p0:self.p0 + 2 * n]


class NTT:
    """MULT-LOGDEPTH session A, unpadded NTT multiplier, L = 2n. Not local."""
    written = True

    def __init__(self, inputs, n, seed=None, dtype=torch.float32):
        p = os.path.join(inputs, "ntt.npz")
        assert sha(p) == SHA["ntt.npz"]
        W, self.cfg = LG.load(p)
        self.W = {k: v.to(dtype) for k, v in (rand_like(W, seed) if seed is not None else W).items()}
        self.written = seed is None
        self.n, self.L, self.dtype = n, 2 * n, dtype
        self.K = 12 * int(round(math.log2(self.L))) + 31
        self.reg = {nm: [2 * i, 2 * i + 1] for nm, i in reg_ntt().items()}
        self.reg["anchor"] = [self.cfg["d_model"] - 2, self.cfg["d_model"] - 1]

    def problems(self, N, rng):
        n = self.n
        a, b = rng.integers(0, 10, (N, n)), rng.integers(0, 10, (N, n))
        b[:, 0] = rng.integers(1, 10, N)             # b nonzero, so changing a always changes the product
        a2 = a.copy()
        for r in range(N):
            i = rng.integers(n)
            a2[r, i] = rng.choice([d for d in range(10) if d != a[r, i]])
        return dict(t=LG.encode(a, b, self.L, self.cfg), t2=LG.encode(a2, b, self.L, self.cfg), y=torch.as_tensor(LG.answer(a, b, self.L)),
                    y2=torch.as_tensor(LG.answer(a2, b, self.L)))

    def x0(self, t):
        return self.W["wte"][t] + self.W["wpe"][:t.shape[1]][None]

    def params(self):
        return list(self.W.values())

    def step(self, x):
        W, cfg = self.W, self.cfg
        N, T, _ = x.shape
        D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
        A = H * DH
        for l in range(cfg["n_layer"]):
            h = F.layer_norm(x, (D,), W[f"h{l}.ln1.w"], W[f"h{l}.ln1.b"], LG.EPS)
            qkv = h @ W[f"h{l}.attn.wqkv"] + W[f"h{l}.attn.bqkv"]
            q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(DH)).softmax(-1)
            x = x + (a @ v).transpose(1, 2).reshape(N, T, A) @ W[f"h{l}.attn.wo"] + W[f"h{l}.attn.bo"]
            h = F.layer_norm(x, (D,), W[f"h{l}.ln2.w"], W[f"h{l}.ln2.b"], LG.EPS)
            x = x + LG.gelu_new(h @ W[f"h{l}.mlp.w1"] + W[f"h{l}.mlp.b1"]) @ W[f"h{l}.mlp.w2"] + W[f"h{l}.mlp.b2"]
        return x

    def read(self, x):
        return LG.readout(self.W, x)[:, 0:2 * self.L:2]


class NGPU:
    """the MULT-LOGDEPTH port of the Neural GPU encoder path, a convolutional cellular automaton. Binary, tape a * b."""

    def __init__(self, inputs, n, run=None, seed=None, dtype=torch.float32):
        job = json.load(open(os.path.join(inputs, "ngpu", "ng-bin20-64-0.0005-rx1-s1.json")))["job"]
        self.job = job
        if seed is not None:
            torch.manual_seed(seed)
        self.m = TG.NGPU(job).to(dtype).eval()
        if run is not None:
            self.m.load_state_dict(torch.load(os.path.join(inputs, "ngpu", run + ".pt"), map_location="cpu", weights_only=True))
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.written = False
        self.n, self.K, self.dtype = n, 2 * n + 1, dtype
        self.reg = {}

    def problems(self, N, rng):
        n = self.n
        a, b = TG.sample_pairs(N, n, n, 2, rng)
        b[:, 0] = 1                                  # b odd, so changing a always changes the product
        a2 = a.copy()
        for r in range(N):
            a2[r, rng.integers(n)] ^= 1
        y, y2 = (TG.product(u, b, 2) for u in (a, a2))
        t = TG.ngpu_batch(a, b, y, "cpu", "mul")[0]
        t2 = TG.ngpu_batch(a2, b, y2, "cpu", "mul")[0]
        return dict(t=t, t2=t2, y=torch.as_tensor(y), y2=torch.as_tensor(y2))

    def x0(self, t):
        s = self.m
        e = s.em(t).permute(0, 2, 1)[:, :, None]
        e = s.nrm(s.inp(e).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return torch.cat([e, torch.zeros(t.shape[0], e.shape[1], s.c["height"] - 1, t.shape[1], dtype=e.dtype)], 2)

    def params(self):
        return list(self.m.parameters())

    def step(self, x):
        g = self.m.sets[0]
        return g[1](g[0](x))

    def read(self, x):
        return self.m.out(x)[:, :, 1].permute(0, 2, 1)[:, :2 * self.n]


def build(c, inputs):
    dtype = torch.float64 if c.get("fp64") else torch.float32
    if c["net"] == "anylen":
        return Anylen(inputs, c["n"], c.get("seed"), dtype)
    if c["net"] == "ntt":
        return NTT(inputs, c["n"], c.get("seed"), dtype)
    return NGPU(inputs, c["n"], c.get("run"), c.get("seed"), dtype)


def pn(x):
    """per-problem L2 norm."""
    return x.flatten(1).norm(dim=1)


def ce_per(lg, y):
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).mean(1)


def m_per(lg, y, y2):
    """sum over digits where the two answers differ of logit[y2] - logit[y]; labels fixed, nothing switches at argmax."""
    a = lg.gather(-1, y2[..., None])[..., 0] - lg.gather(-1, y[..., None])[..., 0]
    return (a * (y != y2).to(a.dtype)).sum(1)


@torch.no_grad()
def trace(S, x):
    xs = [x]
    for _ in range(S.K):
        xs.append(S.step(xs[-1]))
    return xs


@torch.no_grad()
def setup(S, c):
    rng = np.random.default_rng(c["prob_seed"])
    P = S.problems(c["N"], rng)
    xs, xs2 = trace(S, S.x0(P["t"])), trace(S, S.x0(P["t2"]))
    lg, lg2 = S.read(xs[-1]), S.read(xs2[-1])
    yhat, yhat2 = lg.argmax(-1), lg2.argmax(-1)
    return P, xs, xs2, lg, lg2, yhat, yhat2


def grid(K, spec=None):
    base = spec or [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
    return sorted({d for d in base if d <= K} | {K})


def rlist(x, nd=6):
    """floats as short JSON numbers; non-finite become None."""
    return [None if not math.isfinite(v) else float(f"{v:.{nd}g}") for v in np.asarray(x, dtype=np.float64).ravel().tolist()]


def dyn_mask(S, xs):
    st = torch.stack(xs)
    thr = 1e-3 if S.written else 1e-3 * float(xs[0].pow(2).mean().sqrt() + 1e-12)
    return (st - st[0]).abs().amax(0) > thr


def back(S, c):
    """pass-by-pass reverse mode, one pass in memory at a time."""
    P, xs, xs2, lg, lg2, yhat, yhat2 = setup(S, c)
    y, y2 = P["y"], P["y2"]
    lab, lab2 = y, y2                          # true digits for every net, so M is defined even when a random net's argmax doesn't move
    params = S.params()
    for p in params:
        p.requires_grad_(True)
    dyn = dyn_mask(S, xs)
    x = xs[S.K].detach().requires_grad_(True)
    l = S.read(x)
    ce, m = ce_per(l, lab).sum(), m_per(l, lab, lab2).sum()
    gs = torch.autograd.grad(ce, [x] + params, retain_graph=True, allow_unused=True)
    gc, Gw = gs[0], [torch.zeros_like(p) if g is None else g.clone() for p, g in zip(params, gs[1:])]
    gm = torch.autograd.grad(m, x)[0]
    dM = m_per(lg2, lab, lab2) - m_per(lg, lab, lab2)
    ds = set(grid(S.K, c.get("ds")))
    res = dict(K=S.K, written=S.written, N=c["N"], d=[], g_ce=[], g_m=[], g_ce_dyn=[], g_ce_static=[], g_m_dyn=[], g_m_static=[], first=[], delta=[],
               dM=rlist(dM), exact=rlist((yhat == y).all(1).float()), digit=float((yhat == y).float().mean()),
               exact2=rlist((yhat2 == y2).all(1).float()), ce=rlist(ce_per(lg, y)), reg_ce={}, reg_m={})
    keep = {}                                  # partial sums of the per-pass weight gradient, by distance
    wd = []

    def rec(t, gc, gm):
        d = S.K - t
        D = xs2[t] - xs[t]
        first = (gm * D).flatten(1).sum(1).abs() / dM.abs().clamp_min(1e-300)
        res["d"].append(d)
        res["g_ce"].append(rlist(pn(gc)))
        res["g_m"].append(rlist(pn(gm)))
        res["g_ce_dyn"].append(rlist(pn(gc * dyn)))
        res["g_ce_static"].append(rlist(pn(gc * ~dyn)))
        res["g_m_dyn"].append(rlist(pn(gm * dyn)))
        res["g_m_static"].append(rlist(pn(gm * ~dyn)))
        res["first"].append(rlist(first))
        res["delta"].append(rlist(pn(D)))
        if S.written:
            res["reg_ce"][d] = {nm: float(gc[..., ix].pow(2).sum()) for nm, ix in S.reg.items()}
            res["reg_m"][d] = {nm: float(gm[..., ix].pow(2).sum()) for nm, ix in S.reg.items()}

    rec(S.K, gc, gm)
    for t in range(S.K - 1, -1, -1):
        xt = xs[t].detach().requires_grad_(True)
        yt = S.step(xt)
        out = torch.autograd.grad(yt, [xt] + params, grad_outputs=gc, retain_graph=True, allow_unused=True)
        gm = torch.autograd.grad(yt, xt, grad_outputs=gm)[0]
        gc = out[0]
        for a, g in zip(Gw, out[1:]):
            if g is not None:
                a.add_(g)
        d_w = S.K - 1 - t                      # this pass is the d_w-th from the end, 0 = last pass
        if d_w in ds or d_w <= 8:
            keep[d_w] = [a.clone() for a in Gw]
            wd.append(d_w)
        if (S.K - t) in ds:
            rec(t, gc, gm)
    x0 = S.x0(P["t"])
    out = torch.autograd.grad(x0, params, grad_outputs=gc, allow_unused=True)
    Gall = [a + (0 if g is None else g) for a, g in zip(Gw, out)]
    tot = sum(float(a.double().pow(2).sum()) for a in Gall)
    res["w_norm"] = math.sqrt(tot)
    res["w_d"] = wd
    res["w_resid"] = [sum(float((a - b).double().pow(2).sum()) for a, b in zip(Gall, keep[d])) / max(tot, 1e-300) for d in wd]
    return res


def sigma(S, c):
    """top singular value of d(answer logits at K) / d(x_{K-d}): power iteration, J by forward mode one pass at a time, J^T by reverse."""
    P, xs, _, _, _, _, _ = setup(S, c)
    res = dict(K=S.K, N=c["N"], d=[], sigma=[])
    rng = torch.Generator().manual_seed(c["prob_seed"] + 1)
    for d in sorted(set(grid(S.K, c.get("ds")))):
        t = S.K - d
        v = torch.randn(xs[t].shape, generator=rng, dtype=S.dtype)
        v = v / pn(v).view(-1, *[1] * (v.dim() - 1))
        sg = None
        for _ in range(c.get("iters", 16)):
            u = v
            for s in range(t, S.K):
                xc = xs[s].detach()
                _, u = torch.func.jvp(S.step, (xc,), (u,))
            xk = xs[S.K].detach().requires_grad_(True)
            l = S.read(xk)
            _, u_l = torch.func.jvp(S.read, (xs[S.K].detach(),), (u,))
            w = torch.autograd.grad(l, xk, grad_outputs=u_l)[0]
            for s in range(S.K - 1, t - 1, -1):
                xc = xs[s].detach().requires_grad_(True)
                w = torch.autograd.grad(S.step(xc), xc, grad_outputs=w)[0]
            sg = pn(w).sqrt()                  # ||J^T J v|| -> sigma^2 for unit v at convergence
            v = w / pn(w).clamp_min(1e-300).view(-1, *[1] * (w.dim() - 1))
        res["d"].append(d)
        res["sigma"].append(rlist(sg))
    return res


@torch.no_grad()
def run_on(S, x, t, want=None):
    """from state x at pass t to K. want: passes at which to return the state."""
    got = {}
    for s in range(t, S.K):
        x = S.step(x)
        if want is not None and s + 1 in want:
            got[s + 1] = x
    return x, got


def outcome(S, x_end, x_clean_end, size, P, yhat, yhat2):
    lg = S.read(x_end)
    a = lg.argmax(-1)
    return dict(flip=rlist((a != yhat).any(1).float()), tok2=rlist((a == yhat2).all(1).float()), r=rlist(pn(x_end - x_clean_end) / size.clamp_min(1e-300)),
                M=rlist(m_per(lg, P["y"], P["y2"])), digits=rlist((a != yhat).float().sum(1)))


def cf(S, c):
    P, xs, xs2, lg, lg2, yhat, yhat2 = setup(S, c)
    res = dict(K=S.K, N=c["N"], M0=rlist(m_per(lg, P["y"], P["y2"])), M1=rlist(m_per(lg2, P["y"], P["y2"])), rows=[], curves=[], exact=rlist((yhat == P["y"]).all(1).float()),
               exact2=rlist((yhat2 == P["y2"]).all(1).float()))
    for d in grid(S.K, c.get("ds")):
        t = S.K - d
        D = xs2[t] - xs[t]
        size = pn(D)
        for al in c["alphas"]:
            x = xs2[t] if al == 1 else xs[t] + al * D
            want = set(range(t + 1, S.K + 1)) if al in c.get("curve_alphas", ()) else None
            xe, got = run_on(S, x, t, want)
            row = dict(d=d, alpha=al, size=rlist(al * size), **outcome(S, xe, xs[S.K], al * size, P, yhat, yhat2))
            res["rows"].append(row)
            if want:
                res["curves"].append(dict(d=d, alpha=al, s=sorted(got), dist=[rlist(pn(got[s] - xs[s]), 4) for s in sorted(got)]))
    return res


def iso(S, c):
    P, xs, xs2, lg, lg2, yhat, yhat2 = setup(S, c)
    res = dict(K=S.K, N=c["N"], rows=[])
    g = torch.Generator().manual_seed(c["prob_seed"] + 2)
    for d in grid(S.K, c.get("ds")):
        t = S.K - d
        unit = pn(xs2[t] - xs[t])
        for sz in c["sizes"]:
            for dr in range(c.get("draws", 3)):
                e = torch.randn(xs[t].shape, generator=g, dtype=S.dtype)
                e = e / pn(e).view(-1, *[1] * (e.dim() - 1))
                s = sz * unit
                xe, _ = run_on(S, xs[t] + s.view(-1, *[1] * (e.dim() - 1)) * e, t)
                res["rows"].append(dict(d=d, size_rel=sz, draw=dr, size=rlist(s), **outcome(S, xe, xs[S.K], s, P, yhat, yhat2)))
    return res


def grad_dir(S, c):
    P, xs, xs2, lg, lg2, yhat, yhat2 = setup(S, c)
    res = dict(K=S.K, N=c["N"], rows=[])
    x = xs[S.K].detach().requires_grad_(True)
    gm = torch.autograd.grad(m_per(S.read(x), P["y"], P["y2"]).sum(), x)[0]
    want = set(grid(S.K, c.get("ds")))
    dirs = {}
    if 0 in want:
        dirs[0] = gm
    for t in range(S.K - 1, -1, -1):
        xt = xs[t].detach().requires_grad_(True)
        gm = torch.autograd.grad(S.step(xt), xt, grad_outputs=gm)[0]
        if S.K - t in want:
            dirs[S.K - t] = gm
    for d in sorted(dirs):
        t = S.K - d
        unit = pn(xs2[t] - xs[t])
        gn = pn(dirs[d])
        e = dirs[d] / gn.clamp_min(1e-300).view(-1, *[1] * (dirs[d].dim() - 1))
        for sz in c["sizes"]:
            for sign in (1, -1):
                s = sz * unit
                xe, _ = run_on(S, xs[t] + sign * s.view(-1, *[1] * (e.dim() - 1)) * e, t)
                res["rows"].append(dict(d=d, size_rel=sz, sign=sign, gnorm=rlist(gn), size=rlist(s), **outcome(S, xe, xs[S.K], s, P, yhat, yhat2)))
    return res


def reg(S, c):
    P, xs, xs2, lg, lg2, yhat, yhat2 = setup(S, c)
    res = dict(K=S.K, N=c["N"], rows=[])
    rng = np.random.default_rng(c["prob_seed"] + 3)
    for d in grid(S.K, c.get("ds")):
        t = S.K - d
        x = xs[t]
        for nm in c["regs"]:
            ix = S.reg[nm]
            for scope in ("one", "all"):
                # the token: one where this register is nonzero on this problem, else uniform
                mask = torch.ones(x.shape[:2], dtype=x.dtype)
                if scope == "one":
                    mask = torch.zeros(x.shape[:2], dtype=x.dtype)
                    for r in range(x.shape[0]):
                        live = (x[r, :, ix[0]].abs() > 1e-6).nonzero().ravel().tolist()
                        p = live[rng.integers(len(live))] if live else int(rng.integers(x.shape[1]))
                        mask[r, p] = 1
                for mode in (("single", "paired") if len(ix) == 2 else ("single",)):
                    for sz in c["sizes"]:
                        e = torch.zeros_like(x)
                        e[..., ix[0]] = sz * mask
                        if mode == "paired":
                            e[..., ix[1]] = -sz * mask
                        xe, _ = run_on(S, x + e, t)
                        res["rows"].append(dict(d=d, reg=nm, scope=scope, mode=mode, size_reg=sz, **outcome(S, xe, xs[S.K], pn(e), P, yhat, yhat2)))
    return res


PARTS = dict(back=back, sigma=sigma, cf=cf, iso=iso, grad=grad_dir, reg=reg)


def run_one(a):
    c, out, inputs, stop_at = a
    p = os.path.join(out, c["name"] + ".json")
    if os.path.exists(p):
        return c["name"], "skip"
    if time.time() > stop_at:
        return c["name"], "stopped"
    torch.set_num_threads(1)
    t0 = time.time()
    S = build(c, inputs)
    r = PARTS[c["part"]](S, c)
    r.update(case=c, s=round(time.time() - t0, 1))
    with open(p + ".tmp", "w") as f:
        json.dump(r, f)
    os.replace(p + ".tmp", p)
    return c["name"], r["s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--stop-seconds", type=float, default=36000)
    a = ap.parse_args()
    cases = json.load(open(a.cases))
    os.makedirs(a.out, exist_ok=True)
    stop_at = time.time() + a.stop_seconds
    work = [(c, a.out, a.inputs, stop_at) for c in cases]
    if a.procs == 1:
        for w in work:
            print("done", *run_one(w), flush=True)
        return
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for name, s in pool.imap_unordered(run_one, work):
            print("done", name, s, flush=True)


if __name__ == "__main__":
    main()
