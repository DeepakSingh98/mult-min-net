"""Plate: keep training a working network. Written vs trained, from WSP-JOURNEY Experiment 5.

  python3 -B make_followup_train.py RESULTS_DIR

RESULTS_DIR  ../journey/results (c-train/t5 and g-train2/t5 hold one JSON per run)

Plain Adam runs only (lr 1e-6 to 1e-3, 3 seeds x 2 confidence levels per network). Each run stops at its first wrong answer.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # plate.py: the site style helper, copied here
import plate

plate.use_matplotlib()
import matplotlib.pyplot as plt

C = plate.COLORS
OUT = Path(__file__).parent
RES = Path(sys.argv[1])
WRITTEN, TRAINED = ("anylen", "two"), ("ng1", "ng2", "tf")
LRS = (1e-6, 1e-5, 1e-4, 1e-3)
FIRST_N = {"anylen": 4, "two": 4, "ng1": 20, "ng2": 20, "tf": 8}  # length of each net's 256-problem set
LASTED = 3600  # where runs that lasted all 2,000 steps are drawn

runs = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "c-train/t5/*.json")) + glob.glob(str(RES / "g-train2/t5/*.json")))]
runs = [r for r in runs if r["case"]["opt"] == "adam" and "adam_eps" not in r["case"] and "clip" not in r["case"]]
assert len(runs) == 5 * 4 * 6, len(runs)

pts = {("w", lr): [] for lr in LRS} | {("t", lr): [] for lr in LRS}
wrong = {"w": [], "t_small": []}
for r in runs:
    c = r["case"]
    g = "w" if c["net"] in WRITTEN else "t"
    fs = r["fail_step"]
    pts[(g, c["lr"])].append(LASTED if fs is None else fs)
    if fs is not None:
        row = next(h for h in r["hist"] if h["step"] == fs)
        if row["fail_len"] == FIRST_N[c["net"]]:
            k = round(256 * (1 - row["acc"]))
            if g == "w":
                wrong["w"].append(k)
            elif c["lr"] <= 1e-4:
                wrong["t_small"].append(k)
print("written: wrong at first failure", sorted(set(wrong["w"])), "n", len(wrong["w"]))
print("trained, lr <= 1e-4: wrong at first failure", sorted(wrong["t_small"]), "median", np.median(wrong["t_small"]))
for k, v in pts.items():
    print(k, "lasted", sum(x == LASTED for x in v), "/", len(v), "fail steps", sorted(x for x in v if x != LASTED))

fig = plt.figure()
ax = fig.add_axes([0.10, 0.17, 0.60, 0.65])
rng = np.random.default_rng(0)
for i, lr in enumerate(LRS):
    for g, off, col in (("w", -0.19, C["orange"]), ("t", 0.19, C["blue"])):
        ys = np.array(pts[(g, lr)], float)
        xs = i + off + np.linspace(-0.13, 0.13, len(ys))[rng.permutation(len(ys))]
        ys = ys * np.exp(rng.uniform(-0.04, 0.04, len(ys)))  # small vertical jitter so stacked dots show
        ax.scatter(xs, ys, s=90, color=col, lw=0, alpha=0.9, zorder=3)
ax.set_yscale("log")
ax.set_ylim(18, 5200)
ax.set_yticks([25, 100, 500, 2000, LASTED])
ax.set_yticklabels(["25", "100", "500", "2000", "lasted\nall 2000"])
ax.axhline(2600, color=C["axis"], lw=1.5, ls=(0, (4, 4)))
ax.minorticks_off()
ax.set_xlim(-0.6, 3.6)
ax.set_xticks(range(4))
ax.set_xticklabels(["very small", "10×", "100×", "1000×"])
ax.set_xlabel("learning rate")
ax.set_ylabel("steps until the first wrong answer")
ax.grid(axis="x", visible=False)

tx = 0.735
fig.text(tx, 0.78, "Programmed by an LLM", fontsize=23, color=C["orange"], family="serif", va="top")
fig.text(tx, 0.72, "All 256 problems wrong by step 25\nin all 48 runs at every learning rate.",
         fontsize=18, color=C["ink"], linespacing=1.45, va="top")
fig.text(tx, 0.56, "Found by gradient descent", fontsize=23, color=C["blue"], family="serif", va="top")
fig.text(tx, 0.50, "At the smallest rate 15 of 18 last.\nAt 10× and 100× they slip later\nand a slip is usually one problem.\nAt 1000× everything breaks fast.",
         fontsize=18, color=C["ink"], linespacing=1.45, va="top")

fig.text(0.035, 0.925, "Training nudges its own networks and knocks programmed ones over", fontsize=34, family="serif", color=C["ink"])
fig.text(0.035, 0.875, "Keep training a network that gets every problem right. Each dot is one run checked every 25 steps.",
         fontsize=20, color=C["mute"])
fig.text(0.035, 0.04, "Programmed: the any-length and 487-weight multipliers. Trained: two Neural GPUs (binary multiplication) and a looped "
         "transformer (addition).\n3 seeds × 2 confidence levels per network. Fresh Adam each run at 1e-6 to 1e-3. Problems counted on each network's 256-problem test set.", fontsize=17, color=C["mute"], linespacing=1.4)
fig.savefig(OUT / "followup_train.png")
print("wrote followup_train")
