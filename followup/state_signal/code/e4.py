"""WSP-STATE-SIGNAL Experiment 4: does a scaled network survive training? See PROTOCOL.md, the amendment of 2026-11-07 ~09:20 UTC.

  e4.py scan CASES.json OUT --inputs DIR --procs N     matched step scan along fixed directions, no training
  e4.py train CASES.json OUT --inputs DIR --procs N    SGD and Adam runs, stop at first strict failure
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2  # noqa: E402

ALPHA = dict(anylen=5.2, two=6.72)
EPS = [10.0 ** (e / 2) for e in range(-16, 13)]
EMB = ("wte", "wpe")


def start(c, inputs):
    net = e2.Net(c["net"], inputs)
    W = {k: v.clone() for k, v in net.W0.items()}
    if c["cond"] == "scaled":
        W["lnf.w"], W["lnf.b"] = W["lnf.w"] * ALPHA[c["net"]], W["lnf.b"] * ALPHA[c["net"]]
    return net, W


def strict_ok(net, W):
    """(ok, length 4 strict, CE at length 4, failing length). Longer lengths only when shorter ones pass."""
    r = np.random.default_rng(9001)
    t, tg, p0 = net.batch(r, 4, 4, 256)
    with torch.no_grad():
        lg = net.logits(W, t, 4, 4, p0)
    a4 = float((lg.argmax(-1) == tg).all(1).float().mean())
    ce = float(e2.ce(lg, tg))
    if a4 < 1.0 or not math.isfinite(ce):
        return False, a4, ce, 4
    if net.name == "anylen":
        if net.acc(W, np.random.default_rng(9002), 20, 20, 32, bs=32)[0] < 1.0:
            return False, a4, ce, 20
        if net.acc(W, np.random.default_rng(9003), 100, 100, 4, bs=4)[0] < 1.0:
            return False, a4, ce, 100
    return True, a4, ce, None


def unit(d):
    n = math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))
    return {k: v / n for k, v in d.items()}, n


def directions(net, W, seed):
    rng = np.random.default_rng(seed)
    t, tg, p0 = net.batch(rng, 4, 4, 64)
    _, g = e2.grads(net, W, t, tg, 4, 4, p0, "ce", None)
    neg = {k: -v for k, v in g.items()}
    gen = torch.Generator().manual_seed(seed)
    nz = {k: net.W0[k] != 0 for k in W}
    gain = {k: (W[k].clone() if k in ("lnf.w", "lnf.b") else torch.zeros_like(W[k])) for k in W}
    d = dict(grad=neg, grad_noemb={k: (torch.zeros_like(v) if k in EMB else v) for k, v in neg.items()}, grad_nz={k: v * nz[k] for k, v in neg.items()},
             gain=gain, gain_neg={k: -v for k, v in gain.items()}, rand={k: torch.randn(v.shape, generator=gen) for k, v in W.items()},
             rand_nz={k: torch.randn(v.shape, generator=gen) * nz[k] for k, v in W.items()})
    gu, gn = unit(neg)
    gau, _ = unit(gain)
    cos = sum(float((gu[k].double() * gau[k].double()).sum()) for k in W)
    return {k: unit(v)[0] for k, v in d.items()}, gn, cos


def step(W, d, eps):
    return {k: W[k] + (eps * d[k]).to(W[k].dtype) for k in W}


def scan(c, inputs):
    net, W = start(c, inputs)
    ok0 = strict_ok(net, W)
    dirs, gn, cos = directions(net, W, c["grad_seed"])
    res = dict(case=c, base_ok=ok0[0], ce0=ok0[2], g_norm=gn, cos_grad_gain=cos, rows={})
    for name, d in dirs.items():
        line, last, fail, fail_len = [], 0.0, None, None
        for e in EPS:
            ok, a4, ce, fl = strict_ok(net, step(W, d, e))
            line.append([e, a4, ce, ok])
            if not ok:
                fail, fail_len = e, fl
                break
            last = e
        if fail is not None:
            lo, hi = max(last, 1e-12), fail
            for _ in range(8):
                mid = math.sqrt(lo * hi)
                ok, a4, ce, fl = strict_ok(net, step(W, d, mid))
                line.append([mid, a4, ce, ok])
                if ok:
                    lo = mid
                else:
                    hi = mid
                    fail_len = fl
            last = lo
        passed = [x for x in line if x[3]]
        moved = int(sum(int((step(W, d, last)[k] != W[k]).sum()) for k in W)) if last else 0
        res["rows"][name] = dict(eps_star=last, censored=fail is None, fail_len=fail_len, min_ce=min([ok0[2]] + [x[2] for x in passed]), moved_at_star=moved, line=line)
    return res


def train(c, inputs):
    net, W0 = start(c, inputs)
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    W = {k: v.clone().requires_grad_(True) for k, v in W0.items()}
    params = list(W.values())
    frozen = EMB if c["scope"] == "noemb" else ()
    opt = torch.optim.SGD(params, lr=c["lr"]) if c["opt"] == "sgd" else torch.optim.Adam(params, lr=c["lr"])
    pairs = net.pairs()
    disp = lambda: math.sqrt(sum(float((W[k].detach() - W0[k]).double().pow(2).sum()) for k in W))  # noqa: E731
    hist, fail, last_ok = [], None, (0, 0.0, None)
    ok, a4, ce, fl = strict_ok(net, {k: v.detach() for k, v in W.items()})
    last_ok = (0, 0.0, ce)
    tr = []
    r4 = lambda: np.random.default_rng(9001)  # noqa: E731
    for s in range(1, c.get("steps", 2000) + 1):
        n, m = pairs[rng.integers(len(pairs))]
        t, tg, p0 = net.batch(rng, n, m, 64)
        opt.zero_grad()
        l = e2.ce(net.logits(W, t, n, m, p0), tg)
        l.backward()
        for k in frozen:
            if k in W:
                W[k].grad = None
        opt.step()
        tr.append(float(l))
        if s % 25 == 0:
            Wd = {k: v.detach() for k, v in W.items()}
            b4 = r4()
            tt, tgg, pp = net.batch(b4, 4, 4, 256)
            with torch.no_grad():
                lg = net.logits(Wd, tt, 4, 4, pp)
            acc = float((lg.argmax(-1) == tgg).all(1).float().mean())
            cev = float(e2.ce(lg, tgg))
            row = dict(step=s, L4=acc, ce=cev, disp=disp())
            good = acc == 1.0 and math.isfinite(cev)
            if good and s % 100 == 0 and net.name == "anylen":
                row["L20"] = net.acc(Wd, np.random.default_rng(9002), 20, 20, 32, bs=32)[0]
                row["L100"] = net.acc(Wd, np.random.default_rng(9003), 100, 100, 4, bs=4)[0]
                good = row["L20"] == 1.0 and row["L100"] == 1.0
            hist.append(row)
            if good:
                last_ok = (s, row["disp"], cev)
            else:
                fail = s
                break
    return dict(case=c, fail_step=fail, last_ok_step=last_ok[0], disp_ok=last_ok[1], ce_ok=last_ok[2], ce_start=ce, hist=hist[-60:], n_hist=len(hist), loss_first=tr[0] if tr else None)


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
    ap.add_argument("mode", choices=("scan", "train"))
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--stop-seconds", type=float, default=7000)
    a = ap.parse_args()
    cases = json.load(open(a.cases))
    os.makedirs(a.out, exist_ok=True)
    stop_at = time.time() + a.stop_seconds
    work = [(a.mode, c, a.out, a.inputs, stop_at) for c in cases]
    if a.procs == 1:
        for w in work:
            print("done", *run_one(w), flush=True)
        return
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=20) as pool:
        for name, s in pool.imap_unordered(run_one, work, chunksize=1):
            print("done", name, s, flush=True)


if __name__ == "__main__":
    main()
