"""Three plates for the follow-up post, in the site style (style/plate.py).

  python3 -B make_followup_figs.py ANYLEN_DIR NTT_DIR SCAN_DIR

ANYLEN_DIR  MULT-ANYLEN lane dir (arch.py, weights/anylen.npz)
NTT_DIR     MULT-LOGDEPTH lane dir (env/lgarch.py, runs/a1/submissions/04/weights.npz, sha 4a88144218b8b259)
SCAN_DIR    WSP-STATE-SIGNAL results/e4s/scan (Experiment 4 breaking-distance scans)

Each network is run once on one problem here (under a second each on one CPU thread) and checked correct before plotting.
"""
import glob
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # plate.py: the site style helper, copied here
import plate

plate.use_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

torch.set_num_threads(1)
torch.set_grad_enabled(False)
C = plate.COLORS
OUT = Path(__file__).parent
ANYLEN, NTT, SCAN = (Path(p) for p in sys.argv[1:4])


def title(fig, t, sub, foot):
    fig.text(0.035, 0.925, t, fontsize=46, family="serif", color=C["ink"])
    fig.text(0.035, 0.875, sub, fontsize=20, color=C["mute"])
    fig.text(0.035, 0.04, foot, fontsize=18, color=C["mute"])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# 1. The any-length network, pass by pass.
sys.path.insert(0, str(ANYLEN))
arch = importlib.import_module("arch")
cfg, W = arch.load(str(ANYLEN / "weights/anylen.npz"))
rng = np.random.default_rng(7)
n = m = 12
A, B = rng.integers(0, 10, (1, n)), rng.integers(0, 10, (1, m))
A[0, -1], B[0, -1] = max(A[0, -1], 1), max(B[0, -1], 1)
tok, Cd = arch.encode(A, B)
xs = []
lg, halt = arch.forward(W, tok, cfg, every=lambda r, x: xs.append(x[0].clone()))
R = int(halt[0])
s = n + m + 2
assert (lg[0, s:s + n + m].argmax(-1).numpy() == Cd[0]).all()
X = torch.stack(xs[:R]).numpy()
act = np.linalg.norm(np.diff(np.concatenate([W["wte"][tok[0]].numpy()[None], X]), axis=0), axis=-1)  # [R, T]
right = np.stack([(arch.readout(W, torch.from_numpy(x)).argmax(-1).numpy()[s:s + n + m] == Cd[0]) for x in X])  # [R, n+m]
lock = [int(np.flatnonzero(~right[:, k])[-1]) + 1 if (~right[:, k]).any() else 0 for k in range(n + m)]  # 0-based pass index
T = act.shape[1]

fig = plt.figure()
ax = fig.add_axes([0.16, 0.13, 0.81, 0.70])
cmap = LinearSegmentedColormap.from_list("o", [C["cell"], C["orange"]])
v = np.clip((np.log10(act + 1e-4) + 1) / 2.6, 0, 1)  # 0.1 .. 40 -> 0 .. 1
for r in range(R):
    for t in range(T):
        ax.add_patch(plt.Rectangle((r + 0.08, t + 0.08), 0.84, 0.84, color=cmap(v[r, t]), lw=0))
for k, r in enumerate(lock):
    ax.add_patch(plt.Rectangle((r + 0.08, s + k + 0.08), 0.84, 0.84, fill=False, ec=C["ink"], lw=1.6))
ax.set_xlim(0, R)
ax.set_ylim(T, 0)
ax.set_aspect("auto")
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([0.5, 9.5, 19.5, 29.5, 39.5, 49.5, R - 0.5])
ax.set_xticklabels(["1", "10", "20", "30", "40", "50", str(R)])
ax.tick_params(length=0)
ax.set_xlabel("pass")
groups = [(1, n, f"a, {n} digits"), (n + 1, n + 1, "×"), (n + 2, 2 * n + 1, f"b, {m} digits"),
          (s, s, "EOS"), (s, T - 1, f"answer, {n + m} digits")]
ax.set_yticks([])
for a0, a1, lbl in groups:
    ax.text(-0.8, (a0 + a1 + 1) / 2, lbl, ha="right", va="center", fontsize=19, color=C["mute"])
a_int = int("".join(map(str, A[0, ::-1])))
b_int = int("".join(map(str, B[0, ::-1])))
title(fig, "A wave that multiplies",
      f"{a_int:,} × {b_int:,} on the 870-weight network. Each square is how much one token's state changes on one pass.",
      "Outlined: the pass at which each answer digit becomes right and stays right. Brighter = larger change (log scale).")
save(fig, "followup_anylen_wave")

# 2. The NTT network's long-range reads.
sys.path.insert(0, str(NTT / "env"))
lgarch = importlib.import_module("lgarch")
W2, cfg2 = lgarch.load(str(NTT / "runs/a1/submissions/04/weights.npz"))
if "n_layer" not in cfg2:
    W2, cfg2 = cfg2, W2
rng = np.random.default_rng(3)
n2 = 8
L = 16
K = 12 * int(np.log2(L)) + 31
A2, B2 = rng.integers(0, 10, (1, n2)), rng.integers(0, 10, (1, n2))
tok2 = lgarch.encode(A2, B2, L, cfg2)
reads = []
lg2 = lgarch.forward(W2, tok2, cfg2, K, attn=lambda l, h, a: reads.append(a[0, 0].argmax(-1).numpy()) if l == 0 else None)
ok, _ = lgarch.strict(lg2, torch.from_numpy(lgarch.answer(A2, B2, L)))
assert bool(ok[0])
T2 = tok2.shape[1]
off = np.stack(reads) - np.arange(T2)[None]  # [K, T2] read offset of layer 0, head 0
stages = []  # (first pass, last pass, partner of each token)
for r in range(K):
    p = off[r] + np.arange(T2)
    if stages and (stages[-1][2] == p).all():
        stages[-1][1] = r + 1
    else:
        stages.append([r + 1, r + 1, p])
ident = [i for i, st in enumerate(stages) if (st[2] == np.arange(T2)).all()]
stages = stages[:ident[0]]  # the transform; after it, head 0 reads locally (carry)
carry_from = stages[-1][1] + 1
mid = int(np.argmin([np.abs(st[2] - np.arange(T2)).max() for st in stages]))  # the distance-1 stage

fig = plt.figure()
ax = fig.add_axes([0.22, 0.12, 0.76, 0.72])
ys = np.arange(len(stages) + 1)
for i, (r0, r1, p) in enumerate(stages):
    d = int(np.abs(p - np.arange(T2)).max())
    bcast = len(set(p.tolist())) == 1
    col = C["base"] if bcast else (C["orange"] if i <= mid else C["blue"])
    for t in range(T2):
        ax.plot([t, t], [ys[i], ys[i + 1]], color=C["axis"], lw=1.2, zorder=1)
        ax.plot([p[t], t], [ys[i], ys[i + 1]], color=col, lw=2.2, alpha=0.9, zorder=2)
    lbl = f"all read token {p[0]}" if bcast else f"distance {d}"
    ax.text(-1.2, (ys[i] + ys[i + 1]) / 2, f"{lbl}   passes {r0}–{r1}" if r1 > r0 else f"{lbl}   pass {r0}",
            ha="right", va="center", fontsize=17, color=C["mute"])
for y in ys:
    ax.scatter(np.arange(T2), np.full(T2, y), s=26, color=C["ink"], zorder=3, lw=0)
ax.set_xlim(-0.8, T2 - 0.2)
ax.set_ylim(ys[-1] + 0.3, -0.3)
ax.axis("off")
title(fig, "Reading across the whole number",
      f"One attention head of the NTT network, multiplying two {n2}-digit numbers on {T2} tokens. Each line joins a token to the one it reads.",
      f"Orange: forward transform. Blue: inverse transform. Gray: every token reads the last one. Passes {carry_from}–{K}: local carry, not shown.")
save(fig, "followup_ntt_butterfly")

# 3. How far each network can move before an answer goes wrong.
rows = []
for net, label in (("two", "487-weight network"), ("anylen", "any-length network")):
    fs = sorted(glob.glob(str(SCAN / f"scan-{net}-base-*.json")))
    dd = [json.load(open(f)) for f in fs]
    med = {k: float(np.median([d["rows"][k]["eps_star"] for d in dd])) for k in ("grad", "rand", "gain")}
    cens = all(d["rows"]["gain"]["censored"] for d in dd)
    ce0 = float(np.median([d["ce0"] for d in dd]))
    ce_end = float(np.median([d["rows"]["gain"]["min_ce"] for d in dd]))
    rows.append((label, med, cens, ce0, ce_end, len(fs)))
    print(net, med, "gain censored", cens, "ce", ce0, "->", ce_end)

fig = plt.figure()
ax = fig.add_axes([0.20, 0.14, 0.77, 0.68])
dirs = (("grad", "along the gradient", C["orange"]), ("rand", "a random direction", C["base"]),
        ("gain", "the confidence direction", C["blue"]))
y = 0
yt, yl = [], []
for label, med, cens, ce0, ce_end, nf in rows:
    ax.text(-12, y - 0.75, label, ha="left", va="center", fontsize=22, color=C["ink"], family="serif")
    y += 0.25
    for k, dl, col in dirs:
        ax.barh(y, np.log10(med[k]) + 12, left=-12, height=0.72, color=col, lw=0)
        gain = k == "gain" and cens
        txt = f"still right at 1e6, where the scan stops; loss {ce0:.2f} → {ce_end:.2f}" if gain else f"{med[k]:.1e}"
        ax.text(-11.7 if gain else np.log10(med[k]) + 0.25, y, txt, va="center", fontsize=18, color=C["bg"] if gain else C["ink"])
        yt.append(y)
        yl.append(dl)
        y += 1
    y += 1.0
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=17)
ax.set_ylim(y - 0.6, -1.4)
ax.set_xlim(-12, 7)
ax.set_xticks(range(-12, 9, 3))
ax.set_xticklabels([f"1e{t}" for t in range(-12, 9, 3)])
ax.set_xlabel("weight-space distance before the first wrong answer (log scale)")
ax.grid(axis="y", visible=False)
title(fig, "The gradient points at the fragile directions",
      "How far the weights can move in a straight line before any test problem goes wrong. Medians over 8 draws.",
      "The confidence direction scales the final layer norm's weights and biases together: same answers, lower loss.")
save(fig, "followup_break_distance")
