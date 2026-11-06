"""Strict any-length check of a frozen network on Modal CPU (app manylen-verify).

  python3 -B verify_any.py WEIGHTS.npz --plan main [--tag T] [--seed S]     (seed defaults to os.urandom at launch)

Every job runs this directory's arch.forward (shipped as source, sha256 recorded) on problems of fixed lengths (n, m)
with fp32 weights and with bf16-rounded weights (fp32 computation); 'diag' jobs also run float64 and record how far
the network's internal registers ever drift from integers. A problem is strictly right when, at every one of its
n + m answer positions, the correct token's logit is strictly above all 13 others (ties wrong).

Problem kinds (deterministic from the recorded seed):
  exh   every digit string a of length n and b of length m (leading zeros included); 'proper' = no leading zero
  rand  leading digit 1..9 (n > 1), other digits uniform, numpy default_rng([seed, n, m, part])
  edge  every pair from E(n) x E(m), E(L) = 10^(L-1), 10^(L-1)+1, 2*10^(L-1), 5*10^(L-1), 10^L-1, 10^L-2,
        99..90..01 patterns, 1010.., 9090..  (all 9s and powers of ten included)
Each job saves its result to Modal volume manylen-results/<run>/<key>.json as soon as it finishes (and the client
copies it to results/<run>/); a rerun with the same --tag and --seed resumes. Writes results/verify-<tag>-<seed>.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import modal

LANE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy==1.26.4")
         .env({"PYTHONDONTWRITEBYTECODE": "1"}))
app = modal.App("manylen-verify")
vol = modal.Volume.from_name("manylen-results", create_if_missing=True)
CPUS = 2


def edge_values(L):
    v = {10 ** (L - 1), 10 ** (L - 1) + 1, 2 * 10 ** (L - 1), 5 * 10 ** (L - 1), 10 ** L - 1, 10 ** L - 2,
         int(("10" * L)[:L]), int(("90" * L)[:L]), int("9" * (L - 1) + "1") if L > 1 else 1}
    return sorted(x for x in v if 10 ** (L - 1) <= x < 10 ** L or (L == 1 and 0 <= x < 10))


@app.function(image=image, cpu=float(CPUS), memory=4096, timeout=3600, max_containers=24, volumes={"/res": vol})
def job(arch_src, npz_bytes, spec):
    out = check(arch_src, npz_bytes, spec)
    if spec.get("run"):                     # each finished job is saved at once; a preempted job loses only itself
        d = os.path.join("/res", spec["run"])
        os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, spec["key"] + ".json"), "w"))
        vol.commit()
    return out


def check(arch_src, npz_bytes, spec):
    import io
    import types
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(CPUS)
    arch = types.ModuleType("arch")
    exec(arch_src, arch.__dict__)
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as f:
        cfg = json.loads(str(f["config"]))
        W = {k: torch.from_numpy(f[k].astype(np.float32)) for k in f.files if k != "config"}
    n, m, kind = spec["n"], spec["m"], spec["kind"]

    def digits(x, L):
        return [int(c) for c in str(x).zfill(L)[::-1]]

    if kind == "exh":
        a = np.arange(spec["lo"], spec["hi"])
        bb = np.arange(10 ** m)
        av, bv = np.repeat(a, len(bb)), np.tile(bb, len(a))
        A = np.stack([(av // 10 ** i) % 10 for i in range(n)], 1)
        B = np.stack([(bv // 10 ** i) % 10 for i in range(m)], 1)
    elif kind == "rand":
        rng = np.random.default_rng([spec["seed"], n, m, spec.get("part", 0)])
        N = spec["count"]
        A, B = rng.integers(0, 10, (N, n)), rng.integers(0, 10, (N, m))
        if n > 1:
            A[:, n - 1] = rng.integers(1, 10, N)
        if m > 1:
            B[:, m - 1] = rng.integers(1, 10, N)
    elif kind == "edge":
        ea, eb = edge_values(n), edge_values(m)
        pairs = [(x, y) for x in ea for y in eb][spec.get("lo", 0):spec.get("hi")]
        A = np.array([digits(x, n) for x, _ in pairs])
        B = np.array([digits(y, m) for _, y in pairs])
    else:
        raise ValueError(kind)
    proper = ((A[:, -1] > 0) | (n == 1)) & ((B[:, -1] > 0) | (m == 1))
    T = n + m + 3 + n + m
    kmax = spec.get("kmax", 4 * T)
    ks = [k for k in (2 * T, 2 * T + 1, 3 * T, kmax) if k <= kmax]
    out = {"spec": spec, "T": T, "kmax": kmax, "count": int(len(A)), "halt_expected": 2 * n + 3 * m + 2}
    weights = [("fp32", W), ("bf16", arch.to_bf16(W))]
    if spec.get("float64"):
        weights.append(("fp64", {k: v.double() for k, v in W.items()}))
    bs = spec.get("batch", 1000)
    p0 = n + m + 2
    for name, Wx in weights:
        t0 = time.time()
        r = {"strict_wrong": 0, "argmax_wrong": 0, "never_halted": 0, "min_margin": float("inf"),
             "min_margin_proper": float("inf"), "fixed_point_wrong": 0, "min_margin_after_halt": float("inf"),
             "halt_hist": {}, "fixed_K_wrong": {str(k): 0 for k in ks}, "fixed_K_min_margin":
             {str(k): float("inf") for k in ks}, "examples": [], "equiv_checked": 0}
        pos_min = np.full(n + m, np.inf)
        for i in range(0, len(A), bs):
            t, _ = arch.encode(A[i:i + bs], B[i:i + bs])
            N = t.shape[0]
            tg = t[:, p0 + 1:p0 + 1 + n + m]
            after_eos = torch.arange(T)[None, :] >= p0
            halt = torch.zeros(N, dtype=torch.long)
            st = {"dm_halt": torch.full((N, n + m), float("inf"), dtype=torch.float64),
                  "ok_halt": torch.zeros(N, dtype=torch.bool),
                  "after": torch.full((N,), float("inf"), dtype=torch.float64), "K": {}}

            def cb(rr, x):
                lg = arch.readout(Wx, x)
                busy = ((lg.argmax(-1) == arch.BUSY) & after_eos).any(1)
                cell = lg[:, p0:p0 + n + m]
                corr = cell.gather(-1, tg[..., None])[..., 0]
                dm = (corr - cell.scatter(-1, tg[..., None], float("-inf")).max(-1).values).double()
                newly = (~busy) & (halt == 0)
                halt[newly] = rr
                st["dm_halt"][newly] = dm[newly]
                st["ok_halt"][newly] = (cell.argmax(-1) == tg).all(1)[newly]
                hh = halt > 0
                st["after"][hh] = torch.minimum(st["after"][hh], dm.min(1).values[hh])
                if rr in ks:
                    st["K"][rr] = dm.min(1).values
                if i == 0 and bool(newly[:4].any()):
                    st.setdefault("halt_logits", {})
                    for j in newly[:4].nonzero().flatten().tolist():
                        st["halt_logits"][j] = lg[j].clone()
            arch.forward(Wx, t, cfg, K=kmax, every=cb)
            if i == 0:                      # the tracked halting logits equal arch.forward's own halting mode
                ref, rh = arch.forward(Wx, t[:4], cfg)
                for j, l in st.get("halt_logits", {}).items():
                    assert int(rh[j]) == int(halt[j]) and torch.equal(ref[j], l), (name, j)
                    r["equiv_checked"] += 1
            mm = st["dm_halt"].min(1).values.numpy()
            mm[(halt == 0).numpy()] = -np.inf
            bad = mm <= 0
            r["never_halted"] += int((halt == 0).sum())
            r["strict_wrong"] += int(bad.sum())
            r["argmax_wrong"] += int((~st["ok_halt"]).sum())
            r["min_margin"] = min(r["min_margin"], float(mm.min()))
            pp = proper[i:i + bs]
            if pp.any():
                r["min_margin_proper"] = min(r["min_margin_proper"], float(mm[pp].min()))
            pos_min = np.minimum(pos_min, st["dm_halt"].min(0).values.numpy())
            r["fixed_point_wrong"] += int((st["after"] <= 0).sum())
            r["min_margin_after_halt"] = min(r["min_margin_after_halt"], float(st["after"].min()))
            for k in ks:
                r["fixed_K_wrong"][str(k)] += int((st["K"][k] <= 0).sum())
                r["fixed_K_min_margin"][str(k)] = min(r["fixed_K_min_margin"][str(k)], float(st["K"][k].min()))
            for h, c in zip(*np.unique(halt.numpy(), return_counts=True)):
                r["halt_hist"][str(int(h))] = r["halt_hist"].get(str(int(h)), 0) + int(c)
            for j in np.nonzero(bad)[0][:5]:
                if len(r["examples"]) < 5:
                    r["examples"].append(["".join(map(str, A[i + j][::-1])), "".join(map(str, B[i + j][::-1])),
                                          float(mm[j]), int(halt[j])])
        r["min_margin_by_digit"] = [round(float(x), 7) for x in pos_min]
        r["seconds"] = round(time.time() - t0, 1)
        out[name] = r
    if spec.get("drift"):                   # max |register - nearest integer| over all positions and passes
        C = spec["drift"]
        t, _ = arch.encode(A[:spec["drift_count"]], B[:spec["drift_count"]])
        for name, Wx in weights:
            dev = {k: 0.0 for k in C}

            def every(r, x):
                for k, c in C.items():
                    v = x[..., c].double()
                    dev[k] = max(dev[k], float((v - v.round()).abs().max()))
            xs = t if name != "fp64" else t
            arch.forward(Wx, xs, cfg, every=every)
            out[name]["register_drift"] = dev
    return out


# seconds per problem per pass per weight set, measured by the v2 timing run (results/verify-timing-v2-2473545177.json)
# at T = 83, 403, 803; SLOW is headroom. Only used to size jobs, never to score.
PER_PASS = ((83, 27.7 / (200 * 332)), (403, 73.0 / (10 * 1612)), (803, 183.9 / (4 * 3212)))
SLOW = 1.15
PART_SECONDS = 600                          # target per job, so a preemption costs at most ~10-15 min of work


def est_seconds(n, m, count, kmax=None):
    T = 2 * (n + m) + 3
    (t0, c0), (t1, c1), (t2, c2) = PER_PASS
    c = c0 * T / t0 if T < t0 else c2 * (T / t2) ** 1.7 if T > t2 else (
        c0 + (c1 - c0) * (T - t0) / (t1 - t0) if T < t1 else c1 + (c2 - c1) * (T - t1) / (t2 - t1))
    return 2 * count * (kmax or 4 * T) * c * SLOW


def rand_cell(J, seed, n, m, count, part0=0, **kw):
    """count random problems of n x m digits, split into independently seeded parts of about PART_SECONDS each."""
    parts = max(1, -(-est_seconds(n, m, count, kw.get("kmax")) // PART_SECONDS))
    per = int(-(-count // parts))
    for p in range(int(parts)):
        J.append(dict(kind="rand", n=n, m=m, count=per, seed=seed, part=part0 + p, batch=min(per, 2000), **kw))


def plan_main(seed):
    """core cells first (exhaustive n, m <= 3, then 100k random per n = 4..20), so they land early. Cuts made for the
    $25 cap after the first v2 run was lost to preemptions: n >= 150 cells hold 200 problems (was 1,000) and the 100k
    random cells run to 3T passes (every other cell runs to 4T; the 'cap' cells to 3,212 = 4 T(200x200))."""
    J = []
    for n in (1, 2, 3):                     # exhaustive, all digit strings
        for m in (1, 2, 3):
            parts = max(1, -(-est_seconds(n, m, 10 ** (n + m)) // PART_SECONDS))
            step = int(-(-10 ** n // parts))
            for lo in range(0, 10 ** n, step):
                J.append(dict(kind="exh", n=n, m=m, lo=lo, hi=min(10 ** n, lo + step), batch=5000))
    for n in range(4, 21):                  # >= 100k random per n, equal lengths; run to 3T (covers 2T, 2T+1, 3T)
        rand_cell(J, seed, n, n, 100000, kmax=3 * (4 * n + 3))
    for n, cnt in ((32, 1000), (50, 1000), (100, 1000), (200, 200)):
        rand_cell(J, seed, n, n, cnt)
    for n, m in ((7, 3), (3, 7), (20, 1), (1, 20), (13, 2), (2, 13), (17, 9), (9, 17), (5, 1), (1, 5)):
        rand_cell(J, seed, n, m, 10000)
    for n, m in ((100, 1), (1, 100), (100, 3), (3, 100), (64, 7), (7, 64)):
        rand_cell(J, seed, n, m, 1000)
    for n, m in ((200, 1), (1, 200), (150, 40), (40, 150)):
        rand_cell(J, seed, n, m, 200)
    for n, m in ((1, 1), (3, 3), (2, 7), (10, 10), (20, 20), (20, 1)):   # one universal cap: 4 * T(200x200) passes
        rand_cell(J, seed, n, m, 100, part0=55, kmax=4 * 803)
    cells = [(n, m) for n in list(range(1, 33)) + [40, 50, 64, 100] for m in sorted({1, 2, n // 2 or 1, n})]
    for n, m in cells + [(200, 200), (200, 1)]:
        N = len(edge_values(n)) * len(edge_values(m))
        per = max(1, int(N * PART_SECONDS // est_seconds(n, m, N)))
        for lo in range(0, N, per):
            J.append(dict(kind="edge", n=n, m=m, lo=lo, hi=min(N, lo + per), batch=per))
    return J


def plan_drift(seed, C):
    """float64 next to fp32/bf16, plus the largest distance of every register from an integer over all passes."""
    J = []
    for n, cnt, parts in ((3, 8, 1), (10, 8, 1), (30, 8, 1), (100, 8, 2), (200, 8, 4)):
        for p in range(parts):
            J.append(dict(kind="rand", n=n, m=n, count=cnt // parts, seed=seed, part=99 + p, batch=cnt // parts,
                          float64=True, drift=C, drift_count=cnt // parts))
    return J


def plan_big(seed):
    """beyond the main plan's n <= 200: runs to 2T+1 passes (halting is at about 1.25 T) to keep parts <= ~15 min."""
    J = []
    for n, cnt in ((300, 20), (500, 8)):
        rand_cell(J, seed, n, n, cnt, part0=0, kmax=2 * (4 * n + 3) + 1)
    return J


def plan_top200(seed):
    """800 more n = 200 problems on budget left after main (216 there), to 2T+1 passes; brings n = 200 to >= 1k."""
    J = []
    rand_cell(J, seed, 200, 200, 800, part0=1000, kmax=2 * 803 + 1)
    return J


def key(s):
    return "-".join(f"{k}{s[k]}" for k in ("kind", "n", "m", "part", "lo", "kmax") if k in s)


def main(argv=None):
    """spawns every job not yet saved under the run's volume folder (resume = same --seed and --tag), saves each
    result locally as it arrives, stops spawning nothing new after --max-wall seconds, and writes the combined file
    from every result saved so far."""
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--plan", default="main", choices=("main", "drift", "big", "top200", "time"))
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-wall", type=float, default=None)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little")
    src = open(os.path.join(LANE, "arch.py")).read()
    blob = open(a.weights, "rb").read()
    sys.path.insert(0, LANE)
    import build
    C = {k: build.C[k] for k in ("S_in", "Zu_in", "Zc_in", "ZH_in", "Lsep_in", "Fz_in", "P", "Yu", "Yc", "Ans",
                                  "S", "Zu", "Zc", "Fz", "Edge")}
    jobs = {"main": lambda: plan_main(seed), "drift": lambda: plan_drift(seed, C), "big": lambda: plan_big(seed),
            "top200": lambda: plan_top200(seed), "time": lambda: [dict(kind="rand", n=n, m=n, count=c, seed=seed, part=77, batch=c)
                             for n, c in ((20, 200), (100, 10), (200, 4))]}[a.plan]()
    tag = a.tag or a.plan
    run = f"{tag}-{seed}-{hashlib.sha256(blob).hexdigest()[:12]}"
    for s in jobs:
        s["run"], s["key"] = run, key(s)
    assert len({s["key"] for s in jobs}) == len(jobs)
    est = sum(est_seconds(s["n"], s["m"], s.get("count") or (s.get("hi", 0) - s.get("lo", 0)) * 10 ** s["m"]
                          if s["kind"] != "edge" else s["hi"] - s["lo"], s.get("kmax")) for s in jobs)
    print(f"{len(jobs)} jobs, seed {seed}, run {run}, sizing estimate {est / 3600:.1f} container-hours", flush=True)
    if a.dry:
        return jobs
    os.makedirs(os.path.join(LANE, "results", run), exist_ok=True)
    have = {f[:-5] for f in os.listdir(os.path.join(LANE, "results", run)) if f.endswith(".json")}
    try:
        have |= {e.path.split("/")[-1][:-5] for e in vol.listdir(run)}
    except Exception:
        pass
    todo = [s for s in jobs if s["key"] not in have]
    print(f"{len(jobs) - len(todo)} already saved, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        with modal.enable_output(), app.run():
            for r in job.starmap([(src, blob, s) for s in todo], order_outputs=False, return_exceptions=True):
                if isinstance(r, BaseException):
                    print("job failed:", repr(r)[:300], flush=True)
                    continue
                json.dump(r, open(os.path.join(LANE, "results", run, r["spec"]["key"] + ".json"), "w"))
                bad = sum(r[w]["strict_wrong"] + r[w]["fixed_point_wrong"] for w in ("fp32", "bf16"))
                print(f"{time.time() - t0:.0f}s {r['spec']['key']} wrong {bad}", flush=True)
                if a.max_wall and time.time() - t0 > a.max_wall:
                    print("max wall reached; stopping", flush=True)
                    break
    for e in vol.listdir(run):             # collect anything saved remotely but not received locally
        k = e.path.split("/")[-1]
        lp = os.path.join(LANE, "results", run, k)
        if not os.path.exists(lp):
            open(lp, "wb").write(b"".join(vol.read_file(e.path)))
    res = [json.load(open(os.path.join(LANE, "results", run, s["key"] + ".json"))) for s in jobs
           if os.path.exists(os.path.join(LANE, "results", run, s["key"] + ".json"))]
    out = {"weights": os.path.relpath(a.weights, LANE), "weights_sha256": hashlib.sha256(blob).hexdigest(),
           "arch_sha256": hashlib.sha256(src.encode()).hexdigest(), "seed": seed, "seed_source":
           "argument (resume of an os.urandom seed)" if a.seed is not None else "os.urandom at launch",
           "plan": a.plan, "run": run, "jobs_planned": len(jobs), "jobs_done": len(res),
           "missing": [s["key"] for s in jobs if not os.path.exists(os.path.join(LANE, "results", run,
                                                                                s["key"] + ".json"))],
           "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
           "wall_seconds": round(time.time() - t0, 1), "cpus_per_job": CPUS, "jobs": res}
    path = os.path.join(LANE, "results", f"verify-{tag}-{seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    bad = sum(j[w]["strict_wrong"] for j in res for w in ("fp32", "bf16"))
    print(path, f"{len(res)}/{len(jobs)} jobs, strict wrong total", bad, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
