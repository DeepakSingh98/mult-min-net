"""WSP-STATE-SIGNAL Experiment 3: one TEACH-WRITTEN checkpoint (or the written teacher) in free-running and reference-fed modes.

  e3.py --ckpt DIR/best.npz --name s-hints-1 --arm hints --program P --norm N --out OUT
  e3.py --teacher --name teacher --program P --norm N --out OUT

Modes (see PROTOCOL.md). `lo` and `hi` are per-instance step bounds: hints are the true ones at steps lo <= i < hi.
  F     own soft hints, own hidden (official evaluation)
  R     true hints at every step, own hidden
  Z     true hints at every step, hidden zero at every step
  H25.. true hints up to 25%, 50%, 75% of the instance, then free
  B25.. free up to that step, then true hints
  T     matched arms: true hints plus the teacher's hidden in normalized coordinates at every step
  Rh    teacher only: true hints plus its own free-running hidden
One JSON per set: OUT/NAME-nN-sS.json. A finished file is skipped.
"""
import argparse
import json
import os
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import common as C
import clrs
from clrs._src import decoders
import train_teach as TT

MASKS = ("p", "r", "i", "j")
SETS = {16: 32, 64: 32, 128: 8}
BIG = 1 << 30
H_BINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def net_fn(spec, dims, wiring, probe, cap_hidden, use_thidden):
    def f(features, lo, hi, zero, thidden):
        def force(i):
            return (i >= lo) & (i < hi)

        def inject(i, hidden, coin):
            if use_thidden:
                units, prev = thidden
                return jnp.where(i > 0, hidden.at[..., units].set(prev[jnp.maximum(i - 1, 0)]), hidden)
            return hidden * (1.0 - zero)

        def on_step(i, lat):
            return {"hidden": lat["hidden"]}
        o, h, ex = TT.net(spec, dims, wiring["enc"], wiring["dec"], force=force, inject=inject, probe=probe,
                          on_step=on_step if cap_hidden else None)([features], True, 0, True, False)
        out = dict(o=o)
        if h:
            out["own"] = {k: jnp.argmax(jnp.stack([x[k] for x in h]), -1).astype(jnp.int16) for k in h[0]}
        if probe:
            out["probe"] = {k: jnp.argmax(v, -1).astype(jnp.int16) for k, v in ex["probe"].items()}
        if cap_hidden:
            out["hidden"] = ex["step"]["hidden"]
        return out
    return hk.transform(f)


def truth_of(fb):
    h = {d.name: np.asarray(d.data) for d in fb.features.hints}
    t = {k: np.argmax(h[k], -1) for k in MASKS}
    t["pred_h"] = np.asarray(h["pred_h"], np.int64)
    return t


def events(t, L, keys):
    """label each transition t -> t+1 by replaying the reference hints in index space: scan, child (the next interval follows from current
    p, r and the pivot) or pop (it comes from pending recursion). rule: does the local rule, current hints and keys only, get a pop right?"""
    S, B = t["p"].shape[0] - 1, t["p"].shape[1]
    lab = np.zeros((S, B), np.int8)            # 0 dead, 1 scan, 2 child, 3 pop
    rule = np.zeros((S, B), bool)
    for b in range(B):
        idx = []
        for s in range(L[b]):
            pi = t["pred_h"][s, b]
            first = int(np.where(pi == np.arange(len(pi)))[0][0])
            nxt = {int(pi[v]): v for v in range(len(pi)) if v != first}
            order = [first]
            while order[-1] in nxt:
                order.append(nxt[order[-1]])
            pos = np.empty(len(pi), int)
            pos[order] = np.arange(len(pi))
            idx.append((pos[t["p"][s, b]], pos[t["r"][s, b]], pos[t["i"][s, b]], order))
        k = keys[b]
        for s in range(L[b] - 1):
            p0, r0, q0, order = idx[s]
            p1, r1, _, _ = idx[s + 1]
            if (p0, r0) == (p1, r1):
                lab[s, b] = 1
            elif (p1, r1) == (p0, q0 - 1) or ((p1, r1) == (q0 + 1, r0) and q0 - 1 <= p0):
                lab[s, b] = 2
            else:
                lab[s, b] = 3
                a = k[order]
                lmax = np.maximum.accumulate(a)
                rmin = np.minimum.accumulate(a[::-1])[::-1]
                settled = (a >= np.r_[-np.inf, lmax[:-1]]) & (a <= np.r_[rmin[1:], np.inf])
                j = r0 + 1
                while j < len(a) and settled[j]:
                    j += 1
                e = j
                while e + 1 < len(a) and not settled[e + 1]:
                    e += 1
                rule[s, b] = (j, e) == (p1, r1)
    return lab, rule


def stats(ok, live, ch, last):
    r = dict(all=float(ok[live].mean()) if live.any() else None)
    for nm, m in (("read", live & ~ch), ("step", live & ch), ("last", live & last)):
        r[nm] = float(ok[m].mean()) if m.any() else None
        r["n_" + nm] = int(m.sum())
    return r


def score(pred, t, L):
    S = t["p"].shape[0] - 1
    B = t["p"].shape[1]
    step = np.arange(S)[:, None]
    live = step + 1 < L[None, :]
    last = (step + 1) >= 0.75 * L[None, :]
    res, oks = {}, {}
    for k in MASKS:
        ok = pred[k][:S] == t[k][1:]
        ch = t[k][1:] != t[k][:-1]
        res[k] = stats(ok, live, ch, last)
        oks[k] = ok
    ok = pred["pred_h"][:S] == t["pred_h"][1:]
    ch = t["pred_h"][1:] != t["pred_h"][:-1]
    lv = np.broadcast_to(live[..., None], ok.shape)
    ls = np.broadcast_to(last[..., None], ok.shape)
    res["pred_h"] = stats(ok, lv, ch, ls)
    res["pred_h_strict"] = float(ok.all(-1)[live].mean())
    m4 = sum(oks[k].astype(np.float32) for k in MASKS) / 4
    res["m4"] = dict(all=float(m4[live].mean()), last=float(m4[live & last].mean()) if (live & last).any() else None)
    for nm in ("read", "step", "last"):
        v = [res[k][nm] for k in MASKS if res[k][nm] is not None]
        res["m4"][nm] = float(np.mean(v)) if v else None
    all4 = oks["p"] & oks["r"] & oks["i"] & oks["j"]
    return res, m4, all4, oks, live


def horizon(m4, s_b, live, other=()):
    """mean of m4 on steps i >= s, binned by h = i - s + 1 into (2^(k-1), 2^k]. other: same mask on another mode."""
    out = []
    S, B = m4.shape
    h = np.arange(S)[:, None] + 1 - s_b[None, :]
    for k, hi in enumerate(H_BINS):
        lo = H_BINS[k - 1] if k else 0
        m = (h > lo) & (h <= hi) & live
        if m.any():
            out.append([hi, int(m.sum()), float(m4[m].mean())] + [float(x[m].mean()) for x in other])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--program", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="3,20261107")
    ap.add_argument("--sizes", default="16,64,128")
    a = ap.parse_args()
    algo = "quicksort"
    t00 = time.time()
    spec = C.test_set(algo, 16, 4)[1]
    dims = TT.nb_dims_of(C.test_set(algo, 16, 32)[0])
    P = {k: np.asarray(v) for k, v in np.load(a.program).items()}
    wiring = dict(enc=True, dec=True) if a.teacher else TT.ARMS[a.arm]
    probe = not a.teacher
    tf = net_fn(spec, dims, dict(enc=True, dec=True), False, True, False)
    fb0 = C.test_set(algo, 16, 4)[0]
    B0, N0 = 4, 16
    dummy = (jnp.zeros(B0, jnp.int32), jnp.zeros(B0, jnp.int32), jnp.float32(0.0), None)
    ref = C.flatten(tf.init(jax.random.PRNGKey(0), fb0.features, *dummy))
    tp = C.unflatten({k: P.get(k, np.zeros_like(ref[k])) for k in ref})
    params = tp if a.teacher else C.unflatten({k: v for k, v in np.load(a.ckpt).items()})
    matched = a.arm in ("lat", "lathints", "latanneal")
    norm = TT.load_norm(a.norm, "slog", dict(hidden=1, tri=1, node=1, edge=1, graph=1))
    units = norm["hidden"]["units"]
    plain = net_fn(spec, dims, wiring, probe, False, False)
    apply_p = jax.jit(lambda p, f, lo, hi, z: plain.apply(p, None, f, lo, hi, z, None))
    apply_t = jax.jit(lambda f: tf.apply(tp, None, f, jnp.zeros(f.lengths.shape[0], jnp.int32), jnp.zeros(f.lengths.shape[0], jnp.int32), jnp.float32(0.0), None))
    inj = net_fn(spec, dims, wiring, probe, False, True)
    apply_i = jax.jit(lambda p, f, lo, hi, th: inj.apply(p, None, f, lo, hi, jnp.float32(0.0), (units, th)))
    apply_ri = apply_i
    if a.teacher:
        rh = net_fn(spec, dims, wiring, False, False, True)
        apply_ri = jax.jit(lambda p, f, lo, hi, th: rh.apply(p, None, f, lo, hi, jnp.float32(0.0), (jnp.arange(th.shape[-1]), th)))
    os.makedirs(a.out, exist_ok=True)
    for seed in [int(x) for x in a.sets.split(",")]:
        for n in [int(x) for x in a.sizes.split(",")]:
            path = os.path.join(a.out, f"{a.name}-n{n}-s{seed}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            num = SETS[n]
            fb, _ = C.test_set(algo, n, num, seed=seed)
            fb = C.trim_time(fb)
            L = np.asarray(fb.features.lengths).astype(int)
            t = truth_of(fb)
            keys = np.asarray([d.data for d in fb.features.inputs if d.name == "key"][0])
            lab, rule = events(t, L, keys)
            S = t["p"].shape[0] - 1
            B = len(L)
            res = dict(name=a.name, arm=a.arm or "teacher", n=n, seed=seed, num=num, T=int(L.max()), lengths=L.tolist(), modes={}, device=str(jax.devices()[0]),
                       events=dict(scan=int((lab == 1).sum()), child=int((lab == 2).sum()), pop=int((lab == 3).sum()), rule_hits=int(rule.sum())))
            zeros = np.zeros(B, np.int32)
            bign = np.full(B, BIG, np.int32)

            def rs(frac):
                return np.maximum(1, (frac * (L - 1)).astype(np.int32))
            modes = dict(F=(zeros, zeros, 0.0), R=(zeros, bign, 0.0), Z=(zeros, bign, 1.0))
            if not a.teacher:
                if wiring["enc"]:
                    for fr in (25, 50, 75):
                        modes[f"H{fr}"] = (zeros, rs(fr / 100), 0.0)
                        modes[f"B{fr}"] = (rs(fr / 100), bign, 0.0)
                else:
                    modes = dict(F=modes["F"])
            th_raw = None
            if a.teacher or matched:
                th_raw = np.asarray(jax.device_get(apply_t(fb.features)["hidden"]))
            kept = {}
            for mode in list(modes) + (["T"] if matched else []) + (["Rh"] if a.teacher else []):
                if mode == "T":
                    th = jnp.asarray(jax.device_get(TT.to_target("hidden", jnp.asarray(th_raw), norm)))
                    o = apply_i(params, fb.features, jnp.asarray(zeros), jnp.asarray(bign), th)
                elif mode == "Rh":
                    o = apply_ri(params, fb.features, jnp.asarray(zeros), jnp.asarray(bign), jnp.asarray(th_raw))
                else:
                    lo, hi, z = modes[mode]
                    o = apply_p(params, fb.features, jnp.asarray(lo), jnp.asarray(hi), jnp.float32(z))
                p = jax.device_get(decoders.postprocess(spec, o["o"], sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True))
                sc = clrs.evaluate(fb.outputs, p)
                row = dict(score=float(sc["score"]), exact=float(C.per_instance_exact(fb.outputs, p).mean()))
                for head in ("own", "probe"):
                    if head not in o:
                        continue
                    pred = {k: np.asarray(jax.device_get(v)) for k, v in o[head].items()}
                    r, m4, all4, oks, live = score(pred, t, L)
                    pr = oks["p"] & oks["r"]
                    r["events"] = {nm: dict(pr=float(pr[(lab == v) & live].mean()) if ((lab == v) & live).any() else None,
                                            p=float(oks["p"][(lab == v) & live].mean()) if ((lab == v) & live).any() else None,
                                            r=float(oks["r"][(lab == v) & live].mean()) if ((lab == v) & live).any() else None,
                                            n=int(((lab == v) & live).sum())) for nm, v in (("scan", 1), ("child", 2), ("pop", 3))}
                    row[head] = r
                    kept[(mode, head)] = (m4, all4, live)
                res["modes"][mode] = row
                print(a.name, n, seed, mode, round(row["score"], 4), round(time.time() - t0, 1), flush=True)
            res["curves"] = {}
            for head in ("own", "probe"):
                if ("R", head) not in kept:
                    continue
                c = {}
                mR, aR, live = kept[("R", head)]
                mF, aF, _ = kept[("F", head)]
                cover = np.where(live.sum(1) > 0, (aR * live).sum(1) / np.maximum(live.sum(1), 1), 1.0)
                pred_s = np.cumprod(cover)
                surv = np.logical_or(aF, ~live).cumprod(0)
                act = np.where(live.sum(1) > 0, (surv * live).sum(1) / np.maximum(live.sum(1), 1), 0.0)
                pts = sorted({min(S - 1, (1 << k) - 1) for k in range(0, 12)})
                c["compound"] = dict(step=[i + 1 for i in pts], predicted=[float(pred_s[i]) for i in pts], actual=[float(act[i]) for i in pts])
                for fr in (25, 50, 75):
                    for pre in ("H", "B"):
                        if (f"{pre}{fr}", head) in kept:
                            m, _, _ = kept[(f"{pre}{fr}", head)]
                            c[f"{pre}{fr}"] = horizon(m, rs(fr / 100), live, (mR, mF))
                res["curves"][head] = c
            res["s"] = round(time.time() - t0, 1)
            with open(path + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(path + ".tmp", path)
            print("wrote", path, res["s"], flush=True)
    print("done", a.name, round(time.time() - t00, 1), flush=True)


if __name__ == "__main__":
    main()
