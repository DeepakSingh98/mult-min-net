"""Hand-set any-length multiplication transformer (Claude Opus 5.5). Every weight is set in closed form below: no
training, fitting, optimizer or search over weights/masks, no per-length weights and no length cap.

ALGORITHM (a systolic array; every move is local, every register bounded). One pass = blocks 0 and 1.
  * a-stream S: every pass, each position takes its left neighbour's S. Digits seed S, so after t passes position p
    holds a_(p-1-t). BOS (position 0, which only sees itself) keeps S = 0 and the tail flag Tl = 1, so behind a_0 the
    stream is all zeros flagged as tail. Elements ahead of '*' (b, EOS and answer digits) are junk; the separator
    latch Lsep (a wave launched by '*' that moves with the stream) marks the positions the real stream has reached.
  * b stays put: a b-digit position multiplies its own digit DIG by the stream element passing it, gated by
    Lsep (separator already passed) and not G (a wave launched by EOS that marks the answer region first).
  * partial-sum packets travel at half speed through a two-slot delay line: Z <- left neighbour's Y, Y <- own Z.
    The packet in Y at position p on pass t has index k = 2p - t - (n+3); it meets stream element i at b_j exactly
    when i + j = k, so it collects every product of column k once.
  * carries: the lower packet of each position (Z) hands its pending carry to the upper one (Y), which then holds
    V = units + carry-in + product <= 9 + 9 + 81 = 99 and splits it into V mod 10 and a new pending carry <= 9.
  * readout: packets -1, -2, ... carry a head flag H, set at b_0 (left neighbour is '*') while the stream tail passes.
    When packet -1 reaches EOS (in Z), packet 0 sits in EOS's Y slot; a freeze wave Fz starts at EOS and moves one
    position per pass, so it meets packet k in answer cell k's Y slot, where V mod 10 is captured once (Edge) into
    Cap and moved into Ans. Ans holds DIG + captured digit; the tied readout gives logit_j = 2j(Ans - DIG) - j^2.
  * halting: token BUSY has logit 64 - 128 Fz_prev, so an answer cell shows BUSY until its digit is in Ans; the
    caller stops at the first pass with no BUSY at or after EOS (2n + 3m + 2 passes for the scored sequence).

MECHANICS. Head 0 of block 0 has ALiBi slope 12 ln 2 (weights 4096^-d); head 1 has slope 128 (only the key itself).
pred(R) = own(R) + 4097 (o_shift - o_self) recovers the left neighbour exactly up to a leak below 9/4095 per pass.
Every register is rewritten each pass from a snapped (flat-step) MLP function, so leaks and LayerNorm drifts never
accumulate. A +-2^20 anchor pair makes every LayerNorm linear (sigma = 2^17 at d_model 128); reads subtract an
always-zero reference coordinate so the mean cancels. All weights are bfloat16 numbers.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch  # noqa: E402

LN2 = math.log(2.0)
D, DH, NH = 128, 40, 2
SHIFT, SELF = 12 * LN2, 128.0
NAMES = ["A+", "A-", "ref", "one", "DIG", "isStar", "isEOS", "SQ", "Ans", "Ans_in", "Cap",
         "S", "S_in", "Tl", "Tl_in", "Lsep", "Lsep_in", "Lsep_prev", "G", "G_in", "G_prev", "Fb",
         "Zu", "Zu_in", "Zc", "Zc_in", "ZH", "ZH_in", "Yu", "Yc", "YH", "P",
         "Fz", "Fz_in", "Fz_prev", "Edge"]
C = {nm: i for i, nm in enumerate(NAMES)}
G17, ANCHOR = 2.0 ** 17, 2.0 ** 20
KSTEP, WSTEP, KRELU = 2048.0, 0.125, 64.0


def bf(x):
    m, e = np.frexp(np.float64(x))
    return float(np.ldexp(np.round(m * 256) / 256, e))


def is_bf(x):
    return bf(x) == float(x)


class Attn:
    """Collects own()/pred() linear targets for one block's attention, then emits wqkv/wo."""

    def __init__(self):
        self.cm, self.cs = {}, {}           # register -> {target: coefficient} for the shift and self heads

    def _add(self, d, r, x, v):
        d.setdefault(r, {}).setdefault(x, 0.0)
        d[r][x] += v

    def write(self, x, own=(), pred=(), set_=True):
        for r, a in dict(own).items():
            self._add(self.cs, r, x, a)
        for r, b in dict(pred).items():     # pred(r) = own(r) + 4097 (o_shift - o_self)
            self._add(self.cm, r, x, 4097.0 * b)
            self._add(self.cs, r, x, -4096.0 * b)
        if set_:
            self._add(self.cs, x, x, -1.0)

    def emit(self, W, l):
        A = NH * DH
        wqkv, wo = W[f"h{l}.attn.wqkv"], W[f"h{l}.attn.wo"]
        for h, (coef, scale) in enumerate(((self.cm, 17.0), (self.cs, 1.0))):
            regs = sorted(coef, key=NAMES.index)
            assert len(regs) <= DH, (l, h, len(regs))
            for d, r in enumerate(regs):
                col = 2 * A + h * DH + d
                wqkv[C[r], col] += scale
                wqkv[C["ref"], col] -= scale
                for x, v in coef[r].items():
                    if v:
                        wo[h * DH + d, C[x]] += v / scale


class MLP:
    def __init__(self):
        self.units = []                     # (input coefficients, bias, output coefficients)

    def unit(self, coefs, bias, outs):
        self.units.append((dict(coefs), float(bias), dict(outs)))

    def step(self, coefs, c, outs):         # outs * step(sum coefs x + c), ramp over +-1/8
        s = 1.0 / (2 * KSTEP * WSTEP)
        k = {r: KSTEP * v for r, v in coefs.items()}
        self.unit(k, KSTEP * (c + WSTEP), {o: s * v for o, v in outs.items()})
        self.unit(k, KSTEP * (c - WSTEP), {o: -s * v for o, v in outs.items()})

    def relu(self, coefs, c, outs):
        self.unit({r: KRELU * v for r, v in coefs.items()}, KRELU * c, {o: v / KRELU for o, v in outs.items()})

    def snap9(self, x, out):
        for v in range(1, 10):
            self.step({x: 1}, -v + 0.5, {out: 1})

    def emit(self, W, l):
        w1, b1, w2 = W[f"h{l}.mlp.w1"], W[f"h{l}.mlp.b1"], W[f"h{l}.mlp.w2"]
        for u, (coefs, bias, outs) in enumerate(self.units):
            for r, v in coefs.items():
                w1[C[r], u] += v
                w1[C["ref"], u] -= v
            b1[u] = bf(bias)                # a bias that needs more than 8 bits is split over the constant 'one'
            if b1[u] != bias:
                w1[C["one"], u] = bias - b1[u]
            for o, v in outs.items():
                w2[u, C[o]] += v


def block0():
    at, ml = Attn(), MLP()
    # streams and waves
    at.write("S_in", pred={"S": 1}); at.write("S")
    at.write("Tl_in", pred={"Tl": 1}); at.write("Tl")
    for L in ("Lsep", "G", "Fz"):
        at.write(L + "_in", pred={L: 1}); at.write(L + "_prev", own={L: 1}); at.write(L)
    at.write("Fb", pred={"isStar": 1})
    # packets: Z <- left neighbour's Y, Y <- own Z (its pending carry was already handed over)
    at.write("Zu_in", pred={"Yu": 1}); at.write("Zc_in", pred={"Yc": 1}); at.write("ZH_in", pred={"YH": 1})
    at.write("Yu", own={"Zu": 1}); at.write("Yc"); at.write("YH", own={"ZH": 1})
    for r in ("Zu", "Zc", "ZH", "P", "Edge", "Cap"):
        at.write(r)
    at.write("Ans_in", own={"Ans": 1, "DIG": -1, "Cap": 1}); at.write("Ans", own={"DIG": 1})

    ml.snap9("S_in", "S")
    ml.step({"Tl_in": 1}, -0.5, {"Tl": 1})
    for L in ("Lsep", "G"):
        ml.step({L + "_in": 1, L + "_prev": 1}, -0.5, {L: 1})
    ml.snap9("Zu_in", "Zu"); ml.snap9("Zc_in", "Zc")
    ml.step({"ZH_in": 1}, -0.5, {"ZH": 1})
    ml.step({"Fb": 1, "Tl_in": 1}, -1.5, {"YH": 1})
    trig = {"Fz_in": 2, "isEOS": 1, "ZH_in": 1}
    ml.step(dict(trig, Fz_prev=3), -1.5, {"Fz": 1})
    ml.step(dict(trig, Fz_prev=-3), -1.5, {"Edge": 1})
    ml.snap9("Ans_in", "Ans")
    for v in range(1, 10):                  # P = S * DIG if Lsep_prev and not G_prev, exact for integer inputs
        gate = {"S_in": 10, "Lsep_prev": 100, "G_prev": -100}
        ml.relu(dict(gate, DIG=1), -10 * v - 100, {"P": 1})
        ml.relu(gate, -10 * v - 100, {"P": -1})
    return at, ml


def block1():
    at, ml = Attn(), MLP()
    at.write("Yu", own={"Zc": 1, "P": 1}, set_=False)          # V = units + carry-in + product
    for j in range(1, 10):
        ml.step({"Yu": 1}, -10 * j + 0.5, {"Yc": 1, "Yu": -10})
    ml.relu({"Yu": 1, "Edge": 100}, -100, {"Cap": 1})          # capture V mod 10 (V <= 18 in answer cells)
    ml.step({"Yu": 1, "Edge": 100}, -9.5 - 100, {"Cap": -10})
    return at, ml


def build():
    blocks = [block0(), block1()]
    M = max(len(ml.units) for _, ml in blocks)
    cfg = dict(n_layer=2, n_head=NH, d_head=DH, d_model=D, d_mlp=M,
               alibi=[[SHIFT, SELF], [SELF, SELF]])
    W = {k: np.zeros(s, np.float64) for k, s in arch.shapes(cfg).items()}
    for l, (at, ml) in enumerate(blocks):
        at.emit(W, l)
        ml.emit(W, l)
        for ln, mat in (("ln1", f"h{l}.attn.wqkv"), ("ln2", f"h{l}.mlp.w1")):
            read = np.abs(W[mat]).sum(1) > 0
            read[C["one"]] = False
            W[f"h{l}.{ln}.w"][read] = G17
        W[f"h{l}.ln2.b"][C["one"]] = 1.0
    E = W["wte"]
    E[:arch.BUSY, C["A+"]], E[:arch.BUSY, C["A-"]] = ANCHOR, -ANCHOR     # BUSY is output-only: no anchor
    for j in range(10):
        E[j, C["DIG"]] = E[j, C["S"]] = E[j, C["Ans"]] = j
        E[j, C["SQ"]] = j * j
    for t in (arch.MUL, arch.EOS, arch.BOS):
        E[t, C["SQ"]] = 256
    E[arch.MUL, C["isStar"]] = E[arch.MUL, C["Lsep"]] = 1
    E[arch.EOS, C["isEOS"]] = E[arch.EOS, C["G"]] = 1
    E[arch.BOS, C["Tl"]] = 1
    W["lnf.w"][C["Ans"]], W["lnf.w"][C["DIG"]] = 2 * G17, -2 * G17   # 2 (Ans - DIG)
    W["lnf.b"][C["SQ"]] = -1.0                                        # - j^2
    # BUSY: logit 64 - 128 Fz_prev (the reference coordinate cancels the LayerNorm mean), so BUSY is the argmax
    # until the cell's captured digit has reached Ans, and 64 below the answer afterwards
    E[arch.BUSY, C["SQ"]], E[arch.BUSY, C["Fz_prev"]], E[arch.BUSY, C["ref"]] = -64, -128, 128
    W["lnf.w"][C["Fz_prev"]] = W["lnf.w"][C["ref"]] = G17
    bad = [(k, float(v)) for k, a in W.items() for v in a.ravel() if not is_bf(v)]
    assert not bad, bad[:10]
    return cfg, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}


def main():
    cfg, W = build()
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "weights" / "anylen.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    arch.save(out, cfg, W)
    print(out, "nonzero", sum(int((v != 0).sum()) for v in W.values()), "dense", arch.dense(cfg), "cfg", cfg)


if __name__ == "__main__":
    main()
