"""Summarize results/verify-*.json (all numbers straight from the job records) as a markdown table.

  python3 -B analyze.py results/verify-main2-SEED.json [more.json ...]

Per (kind, n, m) cell and weight set: problems, strictly wrong at the halting pass, problems whose halting pass is not
the expected 2n+3m+2 (or never halted), fixed-point failures (some pass from halt to kmax with margin <= 0), the
minimum strict margin at halt and the minimum over every pass from halt to kmax.
"""
import json
import sys
from collections import defaultdict

INF = float("inf")


def main():
    rows = defaultdict(lambda: {"count": 0, "jobs": 0, "T": 0, "kmax": 0, "halt": 0,
                                **{w: {"wrong": 0, "argmax": 0, "never": 0, "fixed": 0, "badhalt": 0,
                                       "min": INF, "min_proper": INF, "after": INF} for w in ("fp32", "bf16", "fp64")}})
    meta = []
    for path in sys.argv[1:]:
        d = json.load(open(path))
        meta.append(f"`{path}`: seed {d['seed']} ({d['seed_source']}), weights sha256 {d['weights_sha256'][:16]}..., "
                    f"arch sha256 {d['arch_sha256'][:16]}..., {d.get('jobs_done', len(d['jobs']))}/"
                    f"{d.get('jobs_planned', len(d['jobs']))} jobs")
        for j in d["jobs"]:
            s = j["spec"]
            r = rows[(s["kind"] + ("-cap" if s.get("kmax") == 3212 else ""), s["n"], s["m"])]
            r["count"] += j["count"]
            r["jobs"] += 1
            r["T"], r["halt"] = j["T"], j["halt_expected"]
            r["kmax"] = max(r["kmax"], j["kmax"])
            for w in ("fp32", "bf16", "fp64"):
                if w not in j:
                    continue
                x, y = j[w], r[w]
                y["wrong"] += x["strict_wrong"]
                y["argmax"] += x["argmax_wrong"]
                y["never"] += x["never_halted"]
                y["fixed"] += x["fixed_point_wrong"]
                y["badhalt"] += sum(c for h, c in x["halt_hist"].items() if int(h) != j["halt_expected"])
                y["min"] = min(y["min"], x["min_margin"])
                y["min_proper"] = min(y["min_proper"], x["min_margin_proper"])
                y["after"] = min(y["after"], x["min_margin_after_halt"])
    out = ["# Verification summary", ""] + [f"- {m}" for m in meta] + ["",
           "| kind | n | m | T | halt pass | kmax | problems | fp32 wrong | fp32 halt != | fp32 fixed-pt wrong | "
           "fp32 min margin (halt / to kmax) | bf16 wrong | bf16 halt != | bf16 fixed-pt wrong | "
           "bf16 min margin (halt / to kmax) |", "|---" * 15 + "|"]
    order = {"exh": 0, "rand": 1, "rand-cap": 2, "edge": 3}
    tot = defaultdict(int)
    for (k, n, m), r in sorted(rows.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1], kv[0][2])):
        f, b = r["fp32"], r["bf16"]
        out.append(f"| {k} | {n} | {m} | {r['T']} | {r['halt']} | {r['kmax']} | {r['count']:,} | {f['wrong']} | "
                   f"{f['badhalt'] + f['never']} | {f['fixed']} | {f['min']:.6f} / {f['after']:.6f} | {b['wrong']} | "
                   f"{b['badhalt'] + b['never']} | {b['fixed']} | {b['min']:.6f} / {b['after']:.6f} |")
        tot["problems"] += r["count"]
        for w in ("fp32", "bf16"):
            for q in ("wrong", "argmax", "never", "fixed", "badhalt"):
                tot[w + q] += r[w][q]
    mins = {w: (min(r[w]["min"] for r in rows.values()), min(r[w]["after"] for r in rows.values()))
            for w in ("fp32", "bf16")}
    out += ["", f"Total: {tot['problems']:,} problems per weight set. " + "; ".join(
        f"{w}: strict wrong {tot[w + 'wrong']}, argmax wrong {tot[w + 'argmax']}, halted at another pass "
        f"{tot[w + 'badhalt']}, never halted {tot[w + 'never']}, fixed-point wrong {tot[w + 'fixed']}, min margin at "
        f"halt {mins[w][0]:.6f}, min margin halt..kmax {mins[w][1]:.6f}" for w in ("fp32", "bf16")) + "."]
    f64 = [(k, n, m, r["fp64"]) for (k, n, m), r in sorted(rows.items()) if r["fp64"]["min"] < INF]
    if f64:
        out.append("float64: " + ", ".join(f"{n}x{m}: wrong {x['wrong']}, min {x['min']:.9f}" for k, n, m, x in f64))
    print("\n".join(out))


if __name__ == "__main__":
    main()
