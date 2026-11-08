"""analyze_e1.py RESULTS_DIR > results/e1_tables.md: every Experiment 1 number from the raw case JSON, read by the rules frozen in PROTOCOL.md."""
import glob
import json
import os
import sys

import numpy as np


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def boot(x, seed=7, B=2000):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return (None, None)
    g = np.random.default_rng(seed)
    m = np.median(x[g.integers(0, len(x), (B, len(x)))], axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fmt(v):
    return "" if v is None else ("0" if v == 0 else f"{v:.2g}")


def label(c):
    n = c["net"]
    if n == "ngpu":
        return ("ngpu-rand" if "seed" in c else ("ngpu-fail" if c["run"].endswith("s3") else "ngpu"))
    return n + ("-rand" if "seed" in c else "")


def copy_of(c):
    return c.get("seed", c.get("run", "w")[-2:] if "run" in c else "w")


def load(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        c = r["case"]
        rows.setdefault((label(c), c["n"]), {}).setdefault(c["part"] + ("64" if c.get("fp64") else ""), []).append(r)
    return rows


def back_class(g0, gs):
    """gs: list of (d, g) for d > 0; the frozen classes."""
    if g0 is None or g0 == 0:
        return "other"
    far = [g for d, g in gs if d >= 8]
    if all(g <= 1e-3 * g0 for g in far):
        return "vanishes"
    if all(g0 / 10 <= g <= 10 * g0 for d, g in gs):
        return "flat"
    if any(g >= 10 * g0 for d, g in gs):
        return "grows"
    return "other"


def pooled(rs, key, d):
    """per-problem values at distance d pooled over copies."""
    out = []
    for r in rs:
        if d in r["d"]:
            out += r[key][r["d"].index(d)]
    return out


def first_vals(rs, d):
    out = []
    for r in rs:
        if d not in r["d"]:
            continue
        i = r["d"].index(d)
        for v, dm, dl in zip(r["first"][i], r["dM"], r["delta"][i]):
            if dm not in (0, None) and dl not in (0, None):
                out.append(v)
    return out


def alpha_star(r):
    """per problem, smallest alpha that changes the answer, by distance."""
    out = {}
    for row in r["rows"]:
        a = out.setdefault(row["d"], {})
        a.setdefault(row["alpha"], row)
    res = {}
    for d, by in out.items():
        al = sorted(by)
        st = []
        for p in range(r["N"]):
            hit = [x for x in al if by[x]["flip"][p] == 1]
            st.append(hit[0] if hit else None)
        res[d] = st
    return res


def cf_row(rs, d, alpha):
    fl, er, y2, n = [], [], [], 0
    for r in rs:
        for row in r["rows"]:
            if row["d"] == d and row["alpha"] == alpha:
                for f, q, t in zip(row["flip"], row["r"], row["tok2"]):
                    n += 1
                    fl.append(f == 1)
                    y2.append(t == 1)
                    er.append(f == 0 and q is not None and q <= 0.01)
    return (np.mean(fl) if n else None, np.mean(er) if n else None, np.mean(y2) if n else None, n)


def flatness(r):
    """share of alpha intervals on which M moves by under 1% of the straight-line slope from M(x_t) to M(x'_t), per problem, by distance."""
    res = {}
    by = {}
    for row in r["rows"]:
        by.setdefault(row["d"], {})[row["alpha"]] = row["M"]
    al = None
    for d, m in by.items():
        al = sorted(m)
        vals = []
        for p in range(r["N"]):
            s = (r["M1"][p] or 0) - (r["M0"][p] or 0)
            if s == 0:
                continue
            xs = [0.0] + al
            ys = [r["M0"][p]] + [m[a][p] for a in al]
            flat = [abs(ys[i + 1] - ys[i]) < 0.01 * abs(s) * (xs[i + 1] - xs[i]) for i in range(len(al)) if ys[i] is not None and ys[i + 1] is not None]
            vals.append(np.mean(flat))
        res[d] = float(np.mean(vals)) if vals else None
    return res


def tension(rs_back, rs_cf):
    """the frozen read at distances d >= 8 with nonzero Delta."""
    ds = sorted({d for r in rs_back for d in r["d"] if d >= 8})
    Ps = {d: med(first_vals(rs_back, d)) for d in ds}
    Ps = {d: v for d, v in Ps.items() if v is not None}
    if not Ps:
        return "undefined", Ps, None, None, None
    p_ok = all(v <= 0.01 for v in Ps.values())
    p_absent = all(v >= 0.1 for v in Ps.values())
    full = np.mean([cf_row(rs_cf, d, 1)[2] for d in ds if cf_row(rs_cf, d, 1)[3]]) if rs_cf else None
    er = [cf_row(rs_cf, d, 0.02)[1] for d in ds if cf_row(rs_cf, d, 0.02)[3]]
    er = min(er) if er else None
    fl = []
    for d in ds:
        tr = [max([cf_row(rs_cf, d, a)[0] or 0 for a in (0.3, 0.5, 0.7, 0.9, 0.99)])]
        fl += tr
    fl = min(fl) if fl else None
    if p_ok and full == 1 and er is not None and er >= 0.95 and fl is not None and fl >= 0.95:
        return "present", Ps, full, er, fl
    return ("absent" if p_absent else "mixed"), Ps, full, er, fl


def answer_share(rs, alpha, dmin=8):
    """post hoc: at distances >= dmin, for trials that did not flip, |M(alpha) - M(0)| / |M(1) - M(0)|, the share of the real answer change that reaches the logits; and the flip share."""
    v, f = [], []
    for r in rs:
        for row in r["rows"]:
            if row["alpha"] == alpha and row["d"] >= dmin:
                for p in range(r["N"]):
                    full = (r["M1"][p] or 0) - (r["M0"][p] or 0)
                    f.append(row["flip"][p] == 1)
                    if full and row["flip"][p] == 0 and row["M"][p] is not None:
                        v.append(abs(row["M"][p] - r["M0"][p]) / abs(full))
    return (med(v), float(np.mean(f)) if f else None)


def main():
    rows = load(sys.argv[1])
    order = ["anylen", "ntt", "ngpu", "ngpu-fail", "anylen-rand", "ntt-rand", "ngpu-rand"]
    keys = sorted(rows, key=lambda k: (order.index(k[0]), k[1]))
    print("# Experiment 1 tables\n\nGenerated by `analyze_e1.py` from `results/e1*/e1/*.json`. `anylen` and `ngpu` are cellular automata. Medians over problems, pooled over copies.\n")
    print("## Clean accuracy\n\n| net | n | K | copies | strict | digit | CE |\n|---|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        rs = rows[k].get("back", [])
        if rs:
            print(f"| {k[0]} | {k[1]} | {rs[0]['K']} | {len(rs)} | {np.mean([x for r in rs for x in r['exact']]):.3f} | {np.mean([r['digit'] for r in rs]):.3f} | {med([x for r in rs for x in r['ce']]):.3f} |")
    print("\n## Backward sensitivity by distance d to the end (float32)\n")
    print("`g_ce` L2 norm of d(CE)/d(state); `dyn` the part on coordinates that change on the clean run. `P` is the first-order share of the true answer change.\n")
    DS = [0, 1, 2, 3, 4, 8, 16, 32, 64, 96]
    print("| net | n | quantity | " + " | ".join(f"d={d}" for d in DS) + " | d=K | class |\n|---|---:|---|" + "---:|" * (len(DS) + 1) + "---|")
    for k in keys:
        rs = rows[k].get("back", [])
        if not rs:
            continue
        K = rs[0]["K"]
        for q, nm in (("g_ce", "g_ce"), ("g_ce_dyn", "g_ce dyn"), ("g_m_dyn", "g_m dyn")):
            vals = {d: med(pooled(rs, q, d)) for d in sorted({*DS, K})}
            cl = back_class(vals.get(0), [(d, med(pooled(rs, q, d))) for d in rs[0]["d"] if d > 0])
            print(f"| {k[0]} | {k[1]} | {nm} | " + " | ".join(fmt(vals.get(d)) if d <= K else "" for d in DS) + f" | {fmt(vals[K])} | {cl} |")
        vals = {d: med(first_vals(rs, d)) for d in sorted({*DS, K})}
        print(f"| {k[0]} | {k[1]} | P | " + " | ".join(fmt(vals.get(d)) if d <= K else "" for d in DS) + f" | {fmt(vals[K])} | |")
    print("\n## Float64 check on backward numbers\n")
    print("| net | n | quantity | d=0 | d=1 | d=2 | d=4 | d=8 | d=16 | d=K |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        rs = rows[k].get("back64", [])
        if not rs:
            continue
        K = rs[0]["K"]
        for q in ("g_ce", "g_ce_dyn", "g_ce_static"):
            print(f"| {k[0]} | {k[1]} | {q} | " + " | ".join(fmt(med(pooled(rs, q, d))) for d in (0, 1, 2, 4, 8, 16, K)) + " |")
        print(f"| {k[0]} | {k[1]} | P | " + " | ".join(fmt(med(first_vals(rs, d))) for d in (0, 1, 2, 4, 8, 16, K)) + " |")
    print("\n## Where the weight gradient comes from\n")
    print("Weight gradient split by pass (readout counted in the last pass). `resid` is the squared share left after the last d + 1 passes. Effective depth is the smallest count of last passes leaving at most 1%.\n")
    print("| net | n | K | weight grad norm (sum over problems) | resid after last pass | resid after last 2 | resid after last 5 | effective depth |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        rs = rows[k].get("back", [])
        if not rs:
            continue
        for r in rs:
            wd, wr = r["w_d"], r["w_resid"]
            depth = next((d + 1 for d, v in zip(wd, wr) if v <= 0.01), None)
            g = lambda d: fmt(wr[wd.index(d)]) if d in wd else ""  # noqa: E731
            print(f"| {k[0]} | {k[1]} | {r['K']} | {r['w_norm']:.3g} | {g(0)} | {g(1)} | {g(4)} | {depth if depth else '> ' + str(max(wd) + 1)} |")
    print("\n## sigma1: top singular value of d(answer logits at K) / d(state at K - d)\n")
    print("| net | n | " + " | ".join(f"d={d}" for d in (1, 2, 4, 8, 16, 32)) + " | d=K |\n|---|---:|" + "---:|" * 7)
    for k in keys:
        rs = rows[k].get("sigma", [])
        if not rs:
            continue
        K = rs[0]["K"]
        print(f"| {k[0]} | {k[1]} | " + " | ".join(fmt(med(pooled(rs, "sigma", d))) for d in (1, 2, 4, 8, 16, 32, K)) + " |")
    print("\n## Counterfactual interpolation\n")
    print("State moved a share alpha of the way to the same pass of a problem differing in one input digit. `flip` share of problems with a changed answer; `erased` no flip and final distance at most 1% of the initial; `to y'` the answer of the second problem.\n")
    print("| net | n | d | size at alpha 1 | median alpha* | 95% CI | flip at 0.02 | erased at 0.02 | flip at 0.5 | to y' at 0.5 | to y' at 0.99 | to y' at 1 | flat share of path |\n|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        rs = rows[k].get("cf", [])
        if not rs:
            continue
        K = rs[0]["K"]
        for d in [x for x in (1, 4, 16, 32, 64, K) if x <= K]:
            st, fl, sz = [], [], []
            for r in rs:
                s = alpha_star(r).get(d)
                if s is None:
                    continue
                st += [v if v is not None else 2.0 for v in s]
                f = flatness(r).get(d)
                if f is not None:
                    fl.append(f)
                sz += [x for row in r["rows"] if row["d"] == d and row["alpha"] == 1 for x in row["size"]]
            if not st:
                continue
            c = boot(st)
            a2, b2 = cf_row(rs, d, 0.02)[:2]
            a5, _, y5, _ = cf_row(rs, d, 0.5)
            y99, y1 = cf_row(rs, d, 0.99)[2], cf_row(rs, d, 1)[2]
            ms = med(st)
            print(f"| {k[0]} | {k[1]} | {d} | {fmt(med(sz))} | {'none' if ms == 2 else fmt(ms)} | [{fmt(c[0])}, {fmt(c[1])}] | {a2:.3f} | {b2:.3f} | {a5:.3f} | {y5:.3f} | {y99:.3f} | {y1:.3f} | {fmt(np.mean(fl)) if fl else ''} |")
    print("\n## Isotropic and gradient-direction perturbation\n")
    print("Size is a multiple of the size of the real information change at that pass. Share of trials flipped and share erased, over the distance grid d >= 1.\n")
    SZ = [0.001, 0.01, 0.1, 1.0, 10.0]
    print("| net | n | direction | " + " | ".join(f"flip at {s:g}" for s in SZ) + " | " + " | ".join(f"erased at {s:g}" for s in SZ) + " |\n|---|---:|---|" + "---:|" * (2 * len(SZ)))
    for k in keys:
        for part in ("iso", "grad"):
            rs = rows[k].get(part, [])
            if not rs:
                continue
            fl, er = {}, {}
            for r in rs:
                for row in r["rows"]:
                    if row["d"] < 1:
                        continue
                    for s in SZ:
                        if abs(row["size_rel"] - s) < 1e-9:
                            fl.setdefault(s, []).extend(x == 1 for x in row["flip"])
                            er.setdefault(s, []).extend(f == 0 and q is not None and q <= 0.01 for f, q in zip(row["flip"], row["r"]))
            print(f"| {k[0]} | {k[1]} | {part} | " + " | ".join(f"{np.mean(fl[s]):.3f}" if s in fl else "" for s in SZ) + " | " + " | ".join(f"{np.mean(er[s]):.3f}" if s in er else "" for s in SZ) + " |")
    print("\n## Named registers\n")
    print("Written nets. One token, single coordinate unless marked; grid distances d >= 1 pooled. Share of trials that flip the answer, and that are erased, by size in register units.\n")
    for k in keys:
        rs = rows[k].get("reg", [])
        if not rs:
            continue
        rr = rs[0]
        sizes = sorted({row["size_reg"] for row in rr["rows"]})
        print(f"\n### {k[0]} n={k[1]}\n")
        print("| register | scope | mode | " + " | ".join(f"{s:g}" for s in sizes) + " | smallest size flipping any | erased at 0.1 |\n|---|---|---|" + "---:|" * (len(sizes) + 2))
        agg = {}
        for row in rr["rows"]:
            if row["d"] < 1:
                continue
            a = agg.setdefault((row["reg"], row["scope"], row["mode"]), {}).setdefault(row["size_reg"], [[], []])
            a[0].extend(x == 1 for x in row["flip"])
            a[1].extend(f == 0 and q is not None and q <= 0.01 for f, q in zip(row["flip"], row["r"]))
        for key in sorted(agg):
            by = agg[key]
            sm = next((s for s in sizes if s in by and np.mean(by[s][0]) > 0), None)
            print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(f"{np.mean(by[s][0]):.2f}" if s in by else "" for s in sizes) + f" | {'' if sm is None else format(sm, 'g')} | {np.mean(by[0.1][1]) if 0.1 in by else float('nan'):.2f} |")
    print("\n## The frozen tension read\n")
    print("| net | n | read | max median P at d >= 8 | min median P | to y' at alpha 1 | lowest erased share at 0.02 | lowest flip share at alpha < 1 |\n|---|---:|---|---:|---:|---:|---:|---:|")
    for k in keys:
        b, c = rows[k].get("back", []), rows[k].get("cf", [])
        if not b:
            continue
        rd, Ps, full, er, fl = tension(b, c)
        hi = max(Ps.values()) if Ps else None
        lo = min(Ps.values()) if Ps else None
        print(f"| {k[0]} | {k[1]} | **{rd}** | {fmt(hi)} | {fmt(lo)} | {fmt(full)} | {fmt(er)} | {fmt(fl)} |")
    print("\n## Post hoc: small upstream changes at the answer\n\nAdded after the frozen read, because the `erased` clause fails on every net. Trials at d >= 8. `share` is the median share of the real answer change that reaches the logits on trials that did not flip.\n")
    print("| net | n | flip at alpha 0.005 | share at 0.005 | flip at 0.02 | share at 0.02 |\n|---|---:|---:|---:|---:|---:|")
    for k in keys:
        c = rows[k].get("cf", [])
        if c:
            s1, f1 = answer_share(c, 0.005)
            s2, f2 = answer_share(c, 0.02)
            print(f"| {k[0]} | {k[1]} | {fmt(f1)} | {fmt(s1)} | {fmt(f2)} | {fmt(s2)} |")


if __name__ == "__main__":
    main()
