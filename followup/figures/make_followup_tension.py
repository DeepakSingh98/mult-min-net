"""Usage: python3 -B make_followup_tension.py ../state_signal/TABLES.md
P (share of the real answer change a first-order prediction at the state recovers) by passes from the end.
Source: WSP-STATE-SIGNAL TABLES.md, 'Backward sensitivity by distance d to the end (float32)', median P rows."""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))  # plate.py: the site style helper, copied here
import plate
plate.use_matplotlib()
import matplotlib.pyplot as plt

T = Path(sys.argv[1]).read_text().split("## Backward sensitivity")[1].split("\n## ")[0]
cols = [c.strip() for c in [l for l in T.splitlines() if l.startswith("| net")][0].strip("|").split("|")]
dcols = [c for c in cols if c.startswith("d=") and c != "d=K"]
rows = {}
for l in T.splitlines():
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) == len(cols) and c[2] == "P":
        rows[(c[0], int(c[1]))] = {int(k[2:]): float(c[cols.index(k)]) for k in dcols if c[cols.index(k)] not in ("", "0")}

C = plate.COLORS
LABEL_Y = {("ngpu-rand", 64): 40, ("ngpu-fail", 64): 0.008}
series = [  # (key, label, colour, linestyle)
    (("anylen", 50), "programmed any-length", C["orange"], "-"),
    (("ntt", 32), "programmed NTT", C["orange"], "--"),
    (("ngpu", 64), "trained Neural GPU", C["blue"], "-"),
    (("ngpu-fail", 64), "failed Neural GPU seed", C["base"], "--"),
    (("ngpu-rand", 64), "random Neural GPU", C["base"], "-"),
]
fig = plt.figure()
ax = fig.add_axes([0.07, 0.15, 0.70, 0.66])
x = lambda d: np.log2(np.asarray(d) + 1)
for key, lbl, col, ls in series:
    r = rows[key]; ds = sorted(r)
    ax.plot(x(ds), [r[d] for d in ds], color=col, ls=ls, lw=3.5)
    ax.text(x(ds[-1]) + 0.12, LABEL_Y.get(key, r[ds[-1]]), f"{lbl}", color=C["ink"], va="center", fontsize=19)
ax.axhline(1, color=C["ink"], ls=(0, (6, 5)), lw=1.5)
ax.text(x(0) + 0.05, 1e2, "real change: swapping in the other state flips the answer at every distance", color=C["ink"], fontsize=18, va="bottom")
ax.set_yscale("log"); ax.set_ylim(1e-45, 1e3)
ax.set_yticks([1e-40, 1e-30, 1e-20, 1e-10, 1]); ax.set_yticklabels(["1e-40", "1e-30", "1e-20", "1e-10", "1"])
tick = [0, 1, 2, 4, 8, 16, 32, 64]
ax.set_xticks(x(tick)); ax.set_xticklabels([str(t) for t in tick]); ax.set_xlim(x(0) - 0.1, x(96) + 0.1)
ax.set_xlabel("passes from the end"); ax.set_ylabel("share the gradient predicts")
ax.grid(axis="x", visible=False)
fig.text(0.035, 0.925, "The gradient misses what earlier passes decide", fontsize=46, family="serif", color=C["ink"])
fig.text(0.035, 0.875, "Share of the real answer change that the gradient at an earlier state predicts. Median over problems.",
         fontsize=20, color=C["mute"])
fig.text(0.035, 0.04, "float32. any-length n=50, NTT n=32, and Neural GPU n=64. The any-length floor near 1e-42 is a float32 limit and float64 keeps falling.",
         fontsize=18, color=C["mute"])
out = Path(__file__).with_name("followup_tension")
fig.savefig(f"{out}.png"); fig.savefig(f"{out}.svg")
for key, *_ in series:
    print(key, {d: rows[key][d] for d in sorted(rows[key])})
