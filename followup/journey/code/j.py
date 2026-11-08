"""WSP-JOURNEY Experiments 5 and 6. See ../PROTOCOL.md.

  j.py scan  CASES.json OUT --inputs DIR --procs N    eps* along fixed unit directions, no training
  j.py train CASES.json OUT --inputs DIR --procs N    optimizer runs from a working network, stop at the first strict failure
  j.py check --inputs DIR                             start-of-run facts: strict at the start, CE, the low-condition alpha

Nets: anylen, two (written; WSP-STATE-SIGNAL e2.Net code paths), ng1, ng2 (trained Neural GPU), tf (trained looped
transformer, addition), untied (anylen with its last J passes untied, counted from the end).
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
from torch.func import functional_call

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch_anylen as A1  # noqa: E402
import e2  # noqa: E402
import train_gd as TG  # noqa: E402

EPS_GRID = [10.0 ** (k / 2) for k in range(-16, 13)]
HIGH_ALPHA = dict(anylen=5.2, two=6.72, untied=5.2)
LOW_CE = 0.52
J_UNTIED = 22
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False   # exact float32 on the L4, as on CPU
torch.backends.cudnn.allow_tf32 = False


def ce(lg, tg, ls=0.0):
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), label_smoothing=ls)


class Written:
    """anylen and two, exactly the WSP-STATE-SIGNAL Experiment 4 problems (single batches from seeds 9001-9003)."""

    def __init__(self, name, inputs):
        self.name = name
        self.net = e2.Net(name, inputs)
        self.W0 = {k: v.clone() for k, v in self.net.W0.items()}
        self.gain_keys = ("lnf.w", "lnf.b")
        self.strict_set = [(4, 4, 256, 9001), (20, 20, 32, 9002), (100, 100, 4, 9003)] if name == "anylen" else [(4, 4, 256, 9001)]
        self.grad_len = (4, 4)

    def batch(self, rng, n, m, N):
        t, tg, p0 = self.net.batch(rng, n, m, N)
        return (t, tg, n, m, p0)

    def logits(self, W, b, train=False):
        t, tg, n, m, p0 = b
        return self.net.logits(W, t, n, m, p0), tg

    def train_pair(self, rng):
        pairs = self.net.pairs()
        return pairs[rng.integers(len(pairs))]


class Untied(Written):
    """anylen with copies of both blocks for the last J passes. Copy j runs pass K-1-j; earlier passes use the written blocks."""

    def __init__(self, inputs, J=J_UNTIED):
        super().__init__("anylen", inputs)
        self.name = "untied"
        base = self.net.W0
        self.block_keys = [k for k in base if k.startswith("h")]
        self.base = {k: v.clone() for k, v in base.items()}
        W0 = {k: v.clone() for k, v in base.items() if not k.startswith("h")}
        for j in range(J):
            for k in self.block_keys:
                W0[f"c{j}.{k}"] = base[k].clone()
        self.W0, self.J = W0, J

    def forward(self, W, tokens, n, m):
        cfg = self.net.cfg
        N, T = tokens.shape
        D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
        A = H * DH
        x = W["wte"][tokens]
        pos = torch.arange(T)
        dist = (pos[:, None] - pos[None, :]).to(x.dtype)
        bias = [(-torch.tensor(cfg["alibi"][l], dtype=x.dtype)[:, None, None] * dist).masked_fill(dist < 0, float("-inf"))
                for l in range(cfg["n_layer"])]
        K = self.net.K(n, m)
        for r in range(K):
            j = K - 1 - r
            P = (lambda k, j=j: W[f"c{j}.{k}"]) if j < self.J else (lambda k: self.base[k])
            for l in range(cfg["n_layer"]):
                h = F.layer_norm(x, (D,), P(f"h{l}.ln1.w"), P(f"h{l}.ln1.b"), A1.EPS)
                qkv = h @ P(f"h{l}.attn.wqkv") + P(f"h{l}.attn.bqkv")
                q, k_, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
                s = (q @ k_.transpose(-1, -2)) / math.sqrt(DH) + bias[l]
                z = s.softmax(-1) @ v
                x = x + z.transpose(1, 2).reshape(N, T, A) @ P(f"h{l}.attn.wo") + P(f"h{l}.attn.bo")
                h = F.layer_norm(x, (D,), P(f"h{l}.ln2.w"), P(f"h{l}.ln2.b"), A1.EPS)
                x = x + A1.gelu_new(h @ P(f"h{l}.mlp.w1") + P(f"h{l}.mlp.b1")) @ P(f"h{l}.mlp.w2") + P(f"h{l}.mlp.b2")
        h = F.layer_norm(x, (D,), W["lnf.w"], W["lnf.b"], A1.EPS)
        return h @ W["wte"].T

    def logits(self, W, b, train=False):
        t, tg, n, m, p0 = b
        return self.forward(W, t, n, m)[:, p0:p0 + n + m], tg


class Module:
    """trained nets through torch.func.functional_call; W holds parameters only."""

    def logits(self, W, b, train=False):
        self.m.train(train)
        return self.fwd(W, b), b[1]

    def train_pair(self, rng):
        """train_gd.train's length rule with the curriculum at its end (cur = lmax)."""
        L = self.lmax
        if rng.random() < 0.8:
            n, m = int(rng.integers(1, L + 1)), int(rng.integers(1, L + 1))
            if rng.random() < 0.5:
                n = L
        else:
            n, m = int(rng.integers(1, L + 1)), int(rng.integers(1, L + 1))
        return n, m


class NG(Module):
    def __init__(self, name, inputs):
        self.name = name
        run = f"ng-bin20-64-0.0005-rx1-s{name[-1]}"
        self.job = json.load(open(os.path.join(inputs, "ngpu", run + ".json")))["job"]
        self.m = TG.NGPU(self.job).eval()
        self.m.load_state_dict(torch.load(os.path.join(inputs, "ngpu", run + ".pt"), map_location="cpu", weights_only=True))
        self.m.to(DEV)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.W0 = {k: v.detach().clone() for k, v in self.m.named_parameters()}
        self.gain_keys = ("out.weight", "out.bias")
        self.strict_set = [(20, 20, 256, 9001), (8, 8, 128, 9002), (64, 64, 32, 9003)]
        self.grad_len, self.lmax = (20, 20), 20

    def batch(self, rng, n, m, N):
        A, B = TG.sample_pairs(N, n, m, 2, rng)
        t, y, w = TG.ngpu_batch(A, B, TG.product(A, B, 2, "mul"), DEV, "mul")
        return (t, y, w)

    def fwd(self, W, b):
        return functional_call(self.m, W, (b[0],))

    def correct(self, W, b):
        with torch.no_grad():
            p = self.logits(W, b)[0].argmax(-1)
        w = b[2]
        return (p[:, :w] == b[1][:, :w]).all(1)


class TF(Module):
    def __init__(self, name, inputs):
        self.name = name
        self.job = json.load(open(os.path.join(inputs, "tf-add-soft-s1.json")))["job"]
        self.m = TG.Looped(self.job).eval()
        self.m.load_state_dict(torch.load(os.path.join(inputs, "tf-add-soft-s1.pt"), map_location="cpu", weights_only=True))
        self.m.to(DEV)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.W0 = {k: v.detach().clone() for k, v in self.m.named_parameters()}
        self.gain_keys = ("lnf.weight", "lnf.bias")
        self.strict_set = [(8, 8, 256, 9001), (4, 4, 128, 9002)]
        self.grad_len, self.lmax = (8, 8), 8

    def batch(self, rng, n, m, N):
        A, B = TG.sample_pairs(N, n, m, 10, rng)
        tok, e = TG.tf_tokens(A, B, TG.product(A, B, 10, "add"), "add", DEV)
        return (tok, tok[:, e + 1:], e, TG.tf_K(self.job["K_rule"], n, m))

    def fwd(self, W, b):
        tok, _, e, K = b
        return functional_call(self.m, W, (tok[:, :-1], K))[:, e:]

    def correct(self, W, b):
        with torch.no_grad():
            p = self.logits(W, b)[0].argmax(-1)
        return (p == b[1]).all(1)


def prune_strict(net):
    """PROTOCOL.md: a length that fails at the start is dropped from that net's strict set and reported."""
    keep, drop = [], []
    with torch.no_grad():
        for row in net.strict_set:
            n, m, N, seed = row
            b = net.batch(np.random.default_rng(seed), n, m, N)
            (keep if bool(correct(net, net.W0, b).all()) else drop).append(row)
    net.strict_set, net.dropped = keep, drop
    return net


def make(name, inputs):
    if name in ("anylen", "two"):
        return Written(name, inputs)
    if name == "untied":
        return Untied(inputs)
    if name in ("ng1", "ng2"):
        return prune_strict(NG(name, inputs))
    if name == "tf":
        return prune_strict(TF(name, inputs))
    raise ValueError(name)


def loss(net, W, b, ls=0.0, train=False):
    lg, tg = net.logits(W, b, train)
    return ce(lg, tg, ls)


def correct(net, W, b):
    if hasattr(net, "correct"):
        return net.correct(W, b)
    with torch.no_grad():
        lg, tg = net.logits(W, b)
    return (lg.argmax(-1) == tg).all(1)


def strict(net, W, upto=None):
    """(ok, first failing length or None, CE at the first length, accuracy at the first length). Lengths in order, stop at the first failure."""
    first_ce = first_acc = None
    for i, (n, m, N, seed) in enumerate(net.strict_set[:upto]):
        b = net.batch(np.random.default_rng(seed), n, m, N)
        ok = correct(net, W, b)
        acc = float(ok.float().mean())
        if i == 0:
            with torch.no_grad():
                first_ce = float(loss(net, W, b))
            first_acc = acc
            if not math.isfinite(first_ce):
                return False, n, first_ce, acc
        if acc < 1.0:
            return False, n, first_ce, first_acc
    return True, None, first_ce, first_acc


def grad_batch(net, seed):
    return net.batch(np.random.default_rng(seed), *net.grad_len, 64)


def scaled(W, keys, a):
    return {k: (v * a if k in keys else v) for k, v in W.items()}


def low_alpha(net):
    """alpha with plain CE on the seed-7700 gradient batch equal to LOW_CE (CE is monotone in a logit scale when every argmax is right)."""
    b = grad_batch(net, 7700)
    f = lambda a: float(loss(net, scaled(net.W0, net.gain_keys, a), b))  # noqa: E731
    lo, hi = 1e-3, 64.0
    with torch.no_grad():
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if f(mid) > LOW_CE:
                lo = mid
            else:
                hi = mid
        a = math.sqrt(lo * hi)
        return a, f(a), f(1.0)


def start(net, cond, alpha=None):
    if cond == "low" and isinstance(net, Written):
        return {k: v.clone() for k, v in net.W0.items()}, 1.0
    if cond == "high" and isinstance(net, Written):
        a = HIGH_ALPHA[net.name]
        return scaled(net.W0, net.gain_keys, a), a
    if cond == "high":
        return {k: v.clone() for k, v in net.W0.items()}, 1.0
    a = alpha if alpha is not None else low_alpha(net)[0]
    return scaled(net.W0, net.gain_keys, a), a


def norm(d):
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))


def unit(d):
    n = norm(d)
    return {k: v / n for k, v in d.items()}, n


def gradient(net, W, b, ls=0.0):
    Wg = {k: v.clone().requires_grad_(True) for k, v in W.items()}
    loss(net, Wg, b, ls).backward()
    return {k: -Wg[k].grad.detach() for k in W}


def directions(net, W, seed):
    b = grad_batch(net, seed)
    g = gradient(net, W, b)
    gen = torch.Generator().manual_seed(seed)
    d = dict(grad=g, rand={k: torch.randn(v.shape, generator=gen, dtype=v.dtype).to(v.device) for k, v in W.items()},
             gain={k: (W[k].clone() if k in net.gain_keys else torch.zeros_like(W[k])) for k in W})
    if isinstance(net, NG):
        d["own"] = gradient(net, W, b, ls=0.1)
    gu, gn = unit(g)
    gau, _ = unit(d["gain"])
    cos = sum(float((gu[k].double() * gau[k].double()).sum()) for k in W)
    extra = {}
    if isinstance(net, Untied):
        tot = sum(float(v.double().pow(2).sum()) for v in g.values())
        extra["copy_share"] = [sum(float(g[f"c{j}.{k}"].double().pow(2).sum()) for k in net.block_keys) / tot for j in range(net.J)]
        extra["shared_share"] = sum(float(g[k].double().pow(2).sum()) for k in g if not k.startswith("c")) / tot
    return {k: unit(v)[0] for k, v in d.items()}, gn, cos, extra


def step(W, d, eps):
    return {k: W[k] + (eps * d[k]).to(W[k].dtype) for k in W}


def scan(c, inputs):
    net = make(c["net"], inputs)
    W, alpha = start(net, c["cond"], c.get("alpha"))
    with torch.no_grad():
        ok0 = strict(net, W)
    dirs, gn, cos, extra = directions(net, W, c["grad_seed"])
    res = dict(case=c, dropped=getattr(net, "dropped", []), alpha=alpha, base_ok=ok0[0], base_fail_len=ok0[1], ce0=ok0[2], g_norm=gn, w_norm=norm(W), cos_grad_gain=cos, rows={}, **extra)
    for name, d in dirs.items():
        line, last, fail, fail_len = [], 0.0, None, None
        with torch.no_grad():
            for e in EPS_GRID:
                ok, fl, ce_, a = strict(net, step(W, d, e))
                line.append([e, a, ce_, ok, fl])
                if not ok:
                    fail, fail_len = e, fl
                    break
                last = e
            if fail is not None:
                lo, hi = max(last, 1e-14), fail
                for _ in range(8):
                    mid = math.sqrt(lo * hi)
                    ok, fl, ce_, a = strict(net, step(W, d, mid))
                    line.append([mid, a, ce_, ok, fl])
                    if ok:
                        lo = mid
                    else:
                        hi, fail_len = mid, fl
                last = lo
        res["rows"][name] = dict(eps_star=last, censored=fail is None, fail_len=fail_len, line=line)
    return res


def train(c, inputs):
    net = make(c["net"], inputs)
    W0, alpha = start(net, c["cond"], c.get("alpha"))
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    W = {k: v.clone().requires_grad_(True) for k, v in W0.items()}
    params = list(W.values())
    if c["opt"] == "sgd":
        opt = torch.optim.SGD(params, lr=c["lr"])
    else:
        opt = torch.optim.Adam(params, lr=c["lr"], eps=c.get("adam_eps", 1e-8))
    w0n = norm(W0)
    disp = lambda: math.sqrt(sum(float((W[k].detach() - W0[k]).double().pow(2).sum()) for k in W))  # noqa: E731
    with torch.no_grad():
        ok, fl, ce0, _ = strict(net, W0)
    hist, fail, fail_len, last_ok, tr = [], None, None, (0, 0.0, ce0), []
    for s in range(1, c.get("steps", 2000) + 1):
        n, m = net.train_pair(rng)
        b = net.batch(rng, n, m, c.get("batch", 64))
        opt.zero_grad()
        lv = loss(net, W, b, c.get("ls", 0.0), c.get("dropout", False))
        lv.backward()
        if c.get("clip"):
            torch.nn.utils.clip_grad_norm_(params, c["clip"])
        opt.step()
        tr.append(float(lv))
        if s % 25 == 0:
            Wd = {k: v.detach() for k, v in W.items()}
            with torch.no_grad():
                ok, fl, cev, acc = strict(net, Wd, upto=None if s % 100 == 0 else 1)
            row = dict(step=s, acc=acc, ce=cev, disp=disp(), ok=ok, fail_len=fl)
            hist.append(row)
            if ok and math.isfinite(cev):
                last_ok = (s, row["disp"], cev)
            else:
                fail, fail_len = s, fl
                break
    return dict(case=c, dropped=getattr(net, "dropped", []), alpha=alpha, start_ok=strict(net, W0)[0], ce_start=ce0, w0_norm=w0n, fail_step=fail, fail_len=fail_len,
                last_ok_step=last_ok[0], disp_ok=last_ok[1], rel_disp_ok=last_ok[1] / w0n, ce_ok=last_ok[2],
                loss_first=tr[0] if tr else None, loss_last=tr[-1] if tr else None, hist=hist[-80:], n_hist=len(hist))


def check(inputs):
    out = {}
    for name in ("anylen", "two", "ng1", "ng2", "tf", "untied"):
        t0 = time.time()
        net = make(name, inputs)
        with torch.no_grad():
            r = dict(strict=strict(net, net.W0), n_params=sum(v.numel() for v in net.W0.values()), w_norm=norm(net.W0))
            if name in ("ng1", "ng2", "tf"):
                r["low_alpha"] = low_alpha(net)
        r["s"] = round(time.time() - t0, 1)
        out[name] = r
        print(name, json.dumps(r), flush=True)
    return out


def run_one(a):
    mode, c, out, inputs, stop_at = a
    p = os.path.join(out, c["name"] + ".json")
    if os.path.exists(p):
        return c["name"], "skip"
    if time.time() > stop_at:
        return c["name"], "stopped"
    torch.set_num_threads(1)
    t0 = time.time()
    r = (scan if mode == "scan" else train)(c, inputs)
    r["s"] = round(time.time() - t0, 1)
    with open(p + ".tmp", "w") as f:
        json.dump(r, f)
    os.replace(p + ".tmp", p)
    return c["name"], r["s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("scan", "train", "check"))
    ap.add_argument("cases", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--stop-seconds", type=float, default=7000)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.mode == "check":
        r = check(a.inputs)
        path = a.out or a.cases
        if path:
            json.dump(r, open(path, "w"), indent=1)
        return
    cases = json.load(open(a.cases))
    if a.only:
        cases = [c for c in cases if c["name"] in a.only]
    os.makedirs(a.out, exist_ok=True)
    stop_at = time.time() + a.stop_seconds
    work = [(a.mode, c, a.out, a.inputs, stop_at) for c in cases]
    if a.procs == 1:
        for w in work:
            print("done", *run_one(w), flush=True)
        return
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=10) as pool:
        for name, s in pool.imap_unordered(run_one, work, chunksize=1):
            print("done", name, s, flush=True)


if __name__ == "__main__":
    main()
