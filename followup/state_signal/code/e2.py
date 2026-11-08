"""WSP-STATE-SIGNAL Experiment 2: repair a damaged written network with the forward pass fixed. Factors: loss (ce, hinge), backward rule (std, sur), support (all, nz).

  e2.py select CASES.json OUT --inputs DIR --procs N    choose the damage draws that count (rejection sampling on 256 dev problems)
  e2.py diag  CASES.json OUT --inputs DIR --procs N     no optimizer: gradient direction at the damaged point and the line to the clean weights
  e2.py repair CASES.json OUT --inputs DIR --procs N    repair runs (one case = one damage draw, one arm, one lr and tau)

Damage draws are a pure function of (net, cell, draw index). The surrogate changes backward only: gelu_new(u) keeps its exact value and its derivative is replaced by
sigmoid(u / (k tau)), with k the unit's largest absolute input weight at the start of repair. See PROTOCOL.md.
"""
import argparse
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
import arch_fixed as A2  # noqa: E402

GELU_C = math.sqrt(2.0 / math.pi)
CELLS = dict(anylen=dict(n1=("noise", 1e-6), n2=("noise", 1e-5), z1=("zero", 1), z4=("zero", 4), s1=("scale", 1.1)),
             two=dict(n1=("noise", 3e-8), n2=("noise", 1e-6), z1=("zero", 1), z4=("zero", 4), s1=("scale", 1.1)))
DEV_DRAWS, SCORED_DRAWS = 2, 12


class SurGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, scale):
        ctx.save_for_backward(u, scale)
        return A1.gelu_new(u)

    @staticmethod
    def backward(ctx, g):
        u, scale = ctx.saved_tensors
        return g * torch.sigmoid(u / scale), None


class Net:
    def __init__(self, name, inputs):
        self.name = name
        self.arch = A1 if name == "anylen" else A2
        self.cfg, W = self.arch.load(os.path.join(inputs, "anylen.npz" if name == "anylen" else "two_layer.npz"))
        self.W0 = {k: v.clone() for k, v in W.items()}

    def K(self, n, m):
        return 2 * n + 3 * m + 2 if self.name == "anylen" else self.cfg["loop"]

    def pairs(self):
        return [(n, m) for n in range(1, 5) for m in range(1, 5)] if self.name == "anylen" else [(4, 4)]

    def batch(self, rng, n, m, N):
        if self.name == "anylen":
            a, b = rng.integers(0, 10, (N, n)), rng.integers(0, 10, (N, m))
            if n > 1:
                a[:, n - 1] = rng.integers(1, 10, N)
            if m > 1:
                b[:, m - 1] = rng.integers(1, 10, N)
            t, _ = A1.encode(a, b)
            p0 = n + m + 2
            return t, t[:, p0 + 1:p0 + 1 + n + m], p0
        t = A2.encode(rng.integers(1000, 10000, N), rng.integers(1000, 10000, N))
        return t, t[:, A2.ANS0:A2.ANS0 + 8], A2.PRED0

    def forward(self, W, tokens, n, m, sur=None):
        """same arithmetic as arch.forward with K passes. sur: None, or {block: per-unit scale} for the surrogate backward."""
        cfg = self.cfg
        N, T = tokens.shape
        D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
        A = H * DH
        x = W["wte"][tokens]
        pos = torch.arange(T)
        if self.name == "anylen":
            dist = (pos[:, None] - pos[None, :]).to(x.dtype)
            bias = [(-torch.tensor(cfg["alibi"][l], dtype=x.dtype)[:, None, None] * dist).masked_fill(dist < 0, float("-inf"))
                    for l in range(cfg["n_layer"])]
        else:
            x = x + W["wpe"][:T]
            mask = torch.ones(T, T, dtype=torch.bool).triu(1)
        for _ in range(self.K(n, m)):
            for l in range(cfg["n_layer"]):
                h = F.layer_norm(x, (D,), W[f"h{l}.ln1.w"], W[f"h{l}.ln1.b"], A1.EPS)
                qkv = h @ W[f"h{l}.attn.wqkv"] + W[f"h{l}.attn.bqkv"]
                q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
                s = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
                s = s + bias[l] if self.name == "anylen" else s.masked_fill(mask, float("-inf"))
                z = s.softmax(-1) @ v
                x = x + z.transpose(1, 2).reshape(N, T, A) @ W[f"h{l}.attn.wo"] + W[f"h{l}.attn.bo"]
                h = F.layer_norm(x, (D,), W[f"h{l}.ln2.w"], W[f"h{l}.ln2.b"], A1.EPS)
                u = h @ W[f"h{l}.mlp.w1"] + W[f"h{l}.mlp.b1"]
                y = A1.gelu_new(u) if sur is None else SurGelu.apply(u, sur[l].expand_as(u))
                x = x + y @ W[f"h{l}.mlp.w2"] + W[f"h{l}.mlp.b2"]
        h = F.layer_norm(x, (D,), W["lnf.w"], W["lnf.b"], A1.EPS)
        return h @ W["wte"].T

    def logits(self, W, t, n, m, p0, sur=None):
        return self.forward(W, t, n, m, sur)[:, p0:p0 + n + m]

    @torch.no_grad()
    def acc(self, W, rng, n, m, N, bs=64):
        hit, dig, tot = 0, 0, 0
        for lo in range(0, N, bs):
            k = min(bs, N - lo)
            t, tg, p0 = self.batch(rng, n, m, k)
            lg = self.logits(W, t, n, m, p0)
            a = lg.argmax(-1) == tg
            hit += int(a.all(1).sum())
            dig += int(a.sum())
            tot += a.numel()
        return hit / N, dig / tot


def ce(lg, tg):
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1))


def hinge(lg, tg, gamma=0.5):
    c = lg.gather(-1, tg[..., None])[..., 0]
    o = lg.clone()
    o.scatter_(-1, tg[..., None], float("-inf"))
    return F.relu(gamma - (c - o.max(-1).values)).mean()


LOSS = dict(ce=ce, hinge=hinge)


def eligible(net, skip_lnf=True):
    """nonzero entries outside lnf, in fixed order."""
    rows = []
    for k in sorted(net.W0):
        if k.startswith("lnf") and skip_lnf:
            continue
        rows += [(k, int(i)) for i in net.W0[k].reshape(-1).nonzero().ravel().tolist()]
    return rows


def damage(net, cell, draw, sigma_p=None):
    """pure function of (net, cell, draw). Returns (W, touched entries)."""
    seed = (zlib_str(net.name + cell) * 1000003 + draw) % (2 ** 63)
    g = torch.Generator().manual_seed(seed)
    W = {k: v.clone() for k, v in net.W0.items()}
    touched = []
    if cell == "p":
        for k in ("lnf.w", "lnf.b"):
            m = W[k] != 0
            W[k] = W[k] * (1 + sigma_p * torch.randn(W[k].shape, generator=g) * m)
        return W, touched
    kind, v = CELLS[net.name][cell]
    if kind == "noise":
        for k in sorted(W):
            W[k] = W[k] * (1 + v * torch.randn(W[k].shape, generator=g))
        return W, touched
    el = eligible(net)
    ids = torch.randperm(len(el), generator=g)[:(v if kind == "zero" else 1)].tolist()
    for i in ids:
        k, j = el[i]
        W[k].reshape(-1)[j] = 0.0 if kind == "zero" else W[k].reshape(-1)[j] * v
        touched.append((k, j))
    return W, touched


def zlib_str(s):
    import zlib
    return zlib.crc32(s.encode())


def wrong_draws(net, cell, want, sigma_p=None, start=0):
    """draws in stream order that are measurably wrong, strict below 0.95 on 256 dev problems at length 4; returns (draw indices, tried)."""
    out, tried = [], 0
    i = start
    while len(out) < want and tried < want * 40:
        W, _ = damage(net, cell, i, sigma_p)
        a, _ = net.acc(W, np.random.default_rng(12345), 4, 4, 256)
        tried += 1
        if a < 0.95:
            out.append(i)
        i += 1
    return out, tried


def scales(net, W, tau):
    """per block, per unit: k tau, with k the unit's largest absolute input weight (bias excluded) at the start of repair."""
    out = {}
    for l in range(net.cfg["n_layer"]):
        k = W[f"h{l}.mlp.w1"].abs().amax(0).clamp_min(1e-6)
        out[l] = (k * tau).detach()
    return out


def mask_of(W, supp, net, touched=(), cell=None):
    """nz: the program's support (the clean network's nonzero entries, lesioned entries included). dmg: only the entries the damage touched
    (amendment 3, an oracle no learner has): `lnf` nonzero entries in cell p, all nonzero entries in noise cells."""
    if supp == "dmg":
        m = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in W.items()}
        for k, j in touched:
            m[k].reshape(-1)[j] = True
        if cell == "p":
            for k in ("lnf.w", "lnf.b"):
                m[k] = net.W0[k] != 0
        return m
    return {k: (net.W0[k] != 0) if supp == "nz" else torch.ones_like(v, dtype=torch.bool) for k, v in W.items()}


def rel_c(W):
    nzs = {k: float(v[v != 0].pow(2).mean().sqrt()) for k, v in W.items() if (v != 0).any()}
    med = float(np.median(list(nzs.values())))
    return {k: nzs.get(k, med) for k in W}


def grads(net, W, t, tg, n, m, p0, lossname, sur):
    Wg = {k: v.detach().clone().requires_grad_(True) for k, v in W.items()}
    l = LOSS[lossname](net.logits(Wg, t, n, m, p0, sur), tg)
    l.backward()
    return float(l), {k: v.grad.detach() for k, v in Wg.items()}


def diag(c, inputs):
    net = Net(c["net"], inputs)
    W, touched = damage(net, c["cell"], c["draw"], c.get("sigma_p"))
    rng = np.random.default_rng(777 + c["draw"])
    t, tg, p0 = net.batch(rng, 4, 4, 256)
    res = dict(case=c, acc=net.acc(W, np.random.default_rng(1), 4, 4, 256), rows=[], line=[])
    d = {k: (net.W0[k] - W[k]) for k in W}
    dn = math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))
    for lossname in ("ce", "hinge"):
        for rule, tau in (("std", None), ("sur", 0.5)):
            sur = None if tau is None else scales(net, W, tau)
            l, g = grads(net, W, t, tg, 4, 4, p0, lossname, sur)
            gn = math.sqrt(sum(float(v.double().pow(2).sum()) for v in g.values()))
            dot = sum(float((-g[k].double() * d[k].double()).sum()) for k in g)
            sk = [k for k in g]
            nz = {k: net.W0[k] != 0 for k in sk}
            gn_nz = math.sqrt(sum(float(g[k][nz[k]].double().pow(2).sum()) for k in sk))
            dn_nz = math.sqrt(sum(float(d[k][nz[k]].double().pow(2).sum()) for k in sk))
            dot_nz = sum(float((-g[k][nz[k]].double() * d[k][nz[k]].double()).sum()) for k in sk)
            row = dict(loss=lossname, rule=rule, loss_v=l, g=gn, cos=dot / max(gn * dn, 1e-300), cos_nz=dot_nz / max(gn_nz * dn_nz, 1e-300), zero_share=1 - gn_nz ** 2 / max(gn ** 2, 1e-300))
            if touched:
                allg = torch.cat([g[k].reshape(-1).abs().double() for k in sorted(g)])
                ranks = []
                for k, j in touched:
                    ranks.append(int((allg > float(g[k].reshape(-1)[j].abs())).sum()))
                row.update(ranks=ranks, entries=int(allg.numel()))
            res["rows"].append(row)
    for b in [i / 10 for i in range(11)]:
        Wb = {k: W[k] + b * d[k] for k in W} if b < 1 else net.W0
        with torch.no_grad():
            lg = net.logits(Wb, t, 4, 4, p0)
        res["line"].append(dict(beta=b, acc=float((lg.argmax(-1) == tg).all(1).float().mean()), ce=float(ce(lg, tg)), hinge=float(hinge(lg, tg))))
    return res


def repair(c, inputs):
    net = Net(c["net"], inputs)
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    W0, touched = damage(net, c["cell"], c["draw"], c.get("sigma_p"))
    W = {k: v.clone().requires_grad_(True) for k, v in W0.items()}
    params = list(W.values())
    c_k = rel_c(W0)
    mask = mask_of(W0, c["supp"], net, touched, c["cell"])
    sur = None if c["rule"] == "std" else scales(net, W0, c["tau"])
    opt = torch.optim.Adam(params, lr=1.0, betas=(0.9, 0.999), eps=1e-8)
    pairs, steps = net.pairs(), c.get("steps", 1500)
    fin = lambda: np.random.default_rng(c["seed"] + 5000)  # noqa: E731

    def monitor(Wd):
        r = np.random.default_rng(c.get("mon_seed", 424242))
        a4 = net.acc(Wd, r, 4, 4, 256)
        if a4[0] < 1.0:                        # the other length pairs only matter once length 4 is right
            return a4[0], a4[1], 0.0
        al = [net.acc(Wd, r, n, m, 64)[0] for n, m in pairs] if len(pairs) > 1 else [a4[0]]
        return a4[0], a4[1], min(al)

    def snap(Wd):
        return {k: v.detach().clone() for k, v in Wd.items()}

    best, best_key, best_step, first_ok = snap(W), (-1, -1, -1), 0, None
    m0 = monitor(snap(W))
    hist = [[0, *m0, None]]
    best_key = (m0[0], m0[2], m0[1])
    skipped, tr = 0, []
    for step in range(1, steps + 1):
        n, m = pairs[rng.integers(len(pairs))]
        t, tg, p0 = net.batch(rng, n, m, c.get("batch", 64))
        opt.zero_grad()
        l = LOSS[c["loss"]](net.logits(W, t, n, m, p0, sur), tg)
        l.backward()
        finite = math.isfinite(float(l)) and all(torch.isfinite(p.grad).all() for p in params)
        if not finite:
            skipped += 1
            continue
        tr.append(float(l))
        with torch.no_grad():
            for k, v in W.items():
                v.grad[~mask[k]] = 0.0
        before = {k: v.detach().clone() for k, v in W.items()}
        opt.step()
        with torch.no_grad():                  # relative Adam: Adam's unit step is rescaled per tensor, then the mask holds
            for k, v in W.items():
                v.data = before[k] + c["lr"] * c_k[k] * (v.data - before[k])
                v.data[~mask[k]] = before[k][~mask[k]]
        if step % 100 == 0 or step == steps:
            Wd = snap(W)
            mt = monitor(Wd)
            hist.append([step, *mt, float(np.mean(tr[-100:])) if tr else None])
            key = (mt[0], mt[2], mt[1])
            if first_ok is None and key > best_key:
                best, best_key, best_step = Wd, key, step
            if first_ok is None and mt[0] == 1.0 and mt[2] == 1.0:
                first_ok, best, best_key, best_step = step, Wd, key, step
                if c["loss"] == "hinge":       # the hinge gradient is about zero from here; nothing further would move
                    break
    last = snap(W)

    def final(Wd):
        r = dict(L4=net.acc(Wd, fin(), 4, 4, 4096)[0])
        if net.name == "anylen" and r["L4"] == 1.0:      # longer lengths only when the shorter one passed, for cost; restored needs all of them
            r["L20"] = net.acc(Wd, fin(), 20, 20, 256, bs=32)[0]
            if r["L20"] == 1.0:
                r["L100"] = net.acc(Wd, fin(), 100, 100, 32, bs=8)[0]
        r["restored"] = all(v == 1.0 for v in r.values()) and (net.name != "anylen" or "L100" in r)
        return r

    def dist(Wd, sub=None):
        a = sum(float((Wd[k] - net.W0[k]).double().pow(2).sum()) if sub is None else float((Wd[k] - net.W0[k])[net.W0[k] != 0].double().pow(2).sum()) for k in Wd)
        return math.sqrt(a)
    return dict(case=c, touched=touched, kept=final(best), last=final(last), best_step=best_step, first_ok=first_ok, skipped=skipped,
                start=[float(x) for x in m0], hist=hist, dist_start=dist(W0), dist_kept=dist(best), dist_kept_nz=dist(best, 1), dist_start_nz=dist(W0, 1),
                loss_first=tr[0] if tr else None, loss_last=float(np.mean(tr[-50:])) if tr else None)


def select(c, inputs):
    """which damage draws count: the first 2 measurably wrong draws are dev, the next 12 scored. Positive control: smallest sigma with enough wrong draws."""
    net = Net(c["net"], inputs)
    cell, sigma = c["cell"], None
    if cell == "p":
        for sigma in (1e-3, 1e-2, 0.1, 0.3):
            ok, tried = wrong_draws(net, cell, DEV_DRAWS + SCORED_DRAWS, sigma)
            if len(ok) == DEV_DRAWS + SCORED_DRAWS and tried <= 3 * len(ok):
                break
    else:
        ok, tried = wrong_draws(net, cell, DEV_DRAWS + SCORED_DRAWS)
    return dict(case=c, sigma_p=sigma, dev=ok[:DEV_DRAWS], scored=ok[DEV_DRAWS:], tried=tried, found=len(ok))


def run_one(a):
    mode, c, out, inputs, stop_at = a
    p = os.path.join(out, c["name"] + ".json")
    if os.path.exists(p):
        return c["name"], "skip"
    if time.time() > stop_at:
        return c["name"], "stopped"
    torch.set_num_threads(1)
    t0 = time.time()
    r = dict(diag=diag, repair=repair, select=select)[mode](c, inputs)
    r["s"] = round(time.time() - t0, 1)
    with open(p + ".tmp", "w") as f:
        json.dump(r, f)
    os.replace(p + ".tmp", p)
    return c["name"], r["s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("diag", "repair", "select"))
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--stop-seconds", type=float, default=36000)
    a = ap.parse_args()
    cases = json.load(open(a.cases))
    os.makedirs(a.out, exist_ok=True)
    stop_at = time.time() + a.stop_seconds
    work = [(a.mode, c, a.out, a.inputs, stop_at) for c in cases]
    if a.procs == 1:
        for w in work:
            print("done", *run_one(w), flush=True)
        return
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=50) as pool:
        for name, s in pool.imap_unordered(run_one, work, chunksize=1):
            print("done", name, s, flush=True)


if __name__ == "__main__":
    main()
