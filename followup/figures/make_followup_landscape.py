"""Hero plate for the follow-up thread: the loss landscape around a perfect network.

  python3 -B make_followup_landscape.py ANYLEN_DIR

ANYLEN_DIR is the MULT-ANYLEN lane dir (arch.py, weights/anylen.npz). A 2D slice of weight space through the written
network, on 32 random 4 x 4-digit problems (seed 11):
  across: a step of size a along the unit negative cross-entropy gradient (the direction training moves);
  along:  the final layer norm's weights and biases both scaled by f (an exact logit gain: same answers, more confident).
Each grid point is one forward pass (about 35 s on one CPU thread); the grid is cached next to this script.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # plate.py: the site style helper, copied here
import plate

plate.use_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

torch.set_num_threads(1)
C = plate.COLORS
OUT = Path(__file__).parent
CACHE = OUT / "followup_landscape.npz"
AA = np.linspace(-1.2e-5, 1.2e-5, 41)
FF = np.geomspace(0.5, 12, 15)

if CACHE.exists():
    d = np.load(CACHE)
    acc, ce = d["acc"], d["ce"]
else:
    ANYLEN = Path(sys.argv[1])
    sys.path.insert(0, str(ANYLEN))
    arch = importlib.import_module("arch")
    cfg, W = arch.load(str(ANYLEN / "weights/anylen.npz"))
    n, N = 4, 32
    rng = np.random.default_rng(11)
    tok, Cd = arch.encode(rng.integers(0, 10, (N, n)), rng.integers(0, 10, (N, n)))
    Y = torch.from_numpy(Cd)
    s, K = 2 * n + 2, 5 * n + 2
    P = {k: v.clone().requires_grad_(True) for k, v in W.items()}
    L = arch.forward(P, tok, cfg, K=K)[:, s:s + 2 * n]
    F.cross_entropy(L.reshape(-1, 14), Y.reshape(-1)).backward()
    g = {k: -p.grad.detach() for k, p in P.items()}
    gn = torch.sqrt(sum((x ** 2).sum() for x in g.values()))
    u = {k: x / gn for k, x in g.items()}
    torch.set_grad_enabled(False)
    acc = np.zeros((len(FF), len(AA)))
    ce = np.zeros_like(acc)
    for i, f in enumerate(FF):
        for j, a in enumerate(AA):
            Wx = {k: W[k] + float(a) * u[k] for k in W}
            Wx["lnf.w"] = Wx["lnf.w"] + (f - 1) * W["lnf.w"]
            Wx["lnf.b"] = Wx["lnf.b"] + (f - 1) * W["lnf.b"]
            L = arch.forward(Wx, tok, cfg, K=K)[:, s:s + 2 * n]
            acc[i, j] = float((L.argmax(-1) == Y).all(1).float().mean())
            ce[i, j] = float(F.cross_entropy(L.reshape(-1, 14), Y.reshape(-1)))
    np.savez(CACHE, aa=AA, ff=FF, acc=acc, ce=ce)

j0, i0 = int(np.argmin(np.abs(AA))), int(np.argmin(np.abs(np.log(FF))))
assert acc[i0, j0] == 1 and (acc[:, j0] == 1).all()
edge = AA[acc[i0] == 1].max()
X, Yg = np.meshgrid(AA * 1e6, np.log10(FF))
Z = -np.log10(ce)  # height: higher = lower loss

fig = plt.figure()
ax = fig.add_axes([-0.02, -0.06, 1.04, 0.98], projection="3d")
ax.set_facecolor(C["bg"])
glow = LinearSegmentedColormap.from_list("o", ["#a8541c", C["orange"], "#ffd2ad"])
fc = np.empty(Z.shape + (4,))
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        fc[i, j] = glow(np.clip(Z[i, j] / 5, 0, 1)) if acc[i, j] == 1 else plt.matplotlib.colors.to_rgba("#33322d")
ax.plot_surface(X, Yg, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0.5, edgecolor=C["bg"], shade=False,
                antialiased=True)
lift = 0.08
ax.plot(X[i0:, j0], Yg[i0:, j0], Z[i0:, j0] + lift, color=C["blue"], lw=4.5, zorder=10)
jj = slice(j0, len(AA))
ax.plot(X[i0, jj], Yg[i0, jj], Z[i0, jj] + lift, color=C["ink"], lw=4.5, zorder=10)
ax.scatter([X[i0, j0]], [Yg[i0, j0]], [Z[i0, j0] + lift], s=170, color=C["ink"], zorder=11, depthshade=False)
ax.text(X[i0, j0] - 2, Yg[i0, j0] - 0.12, Z[i0, j0] + 0.2, "the programmed\nnetwork", color=C["ink"], fontsize=17, ha="right",
        zorder=12)
ax.text(X[-1, j0] - 1, Yg[-1, j0], Z[-1, j0] + 0.5, "the harmless fix:\nsame answers, loss near 0", color=C["ink"],
        fontsize=17, ha="right", zorder=12)
ax.text(X[i0, -1] + 1.2, Yg[i0, -1] - 0.3, Z[i0, -1] - 0.1, "straight along\nthe gradient", color=C["ink"], fontsize=17, ha="left",
        zorder=12)
ax.view_init(elev=24, azim=-62)
ax.set_box_aspect((1.6, 1.15, 0.85))
ax.set_axis_off()

fig.text(0.035, 0.905, "A perfect network on a knife edge", fontsize=46, family="serif", color=C["ink"])
fig.text(0.035, 0.855, "The loss around a programmed network. Orange: every answer right. Gray: some answer wrong. Higher means lower loss.",
         fontsize=19, color=C["mute"])
fig.text(0.035, 0.05, f"Across: a straight line along the gradient, ±1.2e-5 (answers go wrong past {edge:.0e}). Along: the final layer norm scaled "
         f"×{FF[0]:g} to ×{FF[-1]:g}. 32 random 4×4-digit problems.", fontsize=16, color=C["mute"])
save = OUT / "followup_landscape.png"
fig.savefig(save)
print("edge", edge, "loss at start", ce[i0, j0], "loss at far end of fix", ce[-1, j0])
