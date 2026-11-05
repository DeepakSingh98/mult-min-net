"""Adaptive-depth version of the looped network: correct for every loop count K from 41 to 64, with a busy marker.

Written by Claude Opus 5.5 (not Astra) as explicit, closed-form edits on top of Astra's build/looped.py, which it runs
unchanged. No training, fitting or search: every changed value below is derived in explanations/looped_adaptive.md.

Why the 42-pass network needs exactly 42 passes: its readout is a one-shot overwrite on pass 42. Units 41/42 copy
total + 0.75 into the digit coordinate under the gate 255*(r - 41). Before pass 42 the digit coordinate still holds the
visible previous answer digit. After pass 42 the gate stays open, and unit 42, which reads coordinate 0 but writes
coordinate 1, adds about 255*(r - 41) to coordinate 1 every pass.

The edits:
  1. delete the readout units 41/42;
  2. the tied readout reads (total - digit) on every pass. Every digit token j now also carries j on the total
     coordinate, and the final LayerNorm gains are +2^24 on total and -2^24 on digit, so the input digit cancels
     exactly. The balance column (-j) that the old readout used is removed, and lnf.b[total] = -1 keeps the old
     logit formula 4j(t + 1/4) - 2j^2 + 5/4;
  3. the two modulo-ten hinges read the same (total - digit) pair instead of total minus the zero reference;
  4. the counter's LN2 offset -41 served only the pass-42 gate; it is moved into the clock unit's bias (same
     pre-activation, 2 nonzeros fewer);
  5. busy marker: the '*' token gets codes 13 (digit), 13 (total) and 71.5 (triangle). Its logit is then
     52t - 300.25, which crosses the digit-9 line 36t - 151.75 at t = 9.28125. This is between the largest finished
     value (9.006) and the smallest unfinished one (9.53), so '*' is the argmax exactly where a column still needs a
     subtraction.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("looped", Path(__file__).resolve().parent / "looped.py")
looped = importlib.util.module_from_spec(spec)
spec.loader.exec_module(looped)

CFG = dict(looped.CFG, loop=41, d_mlp=42)   # loop = K_min; any K from 41 to 64 gives the same answers
STAR = 10                                   # token id of '*'


def build():
    W, lay = looped.build()
    dig, bal, tri, tot, ctr = lay["digit"], lay["token_balance"], lay["triangle"], lay["total"], lay["counter"]
    ref = lay["references"][0]
    assert (dig, bal, tri, tot, ctr, ref) == (0, 1, 2, 44, 24, 4), (dig, bal, tri, tot, ctr, ref)
    w1, b1, w2 = W["h0.mlp.w1"], W["h0.mlp.b1"], W["h0.mlp.w2"]
    # the units this edit touches, identified by their weights rather than by position alone
    assert w1[tot, 41] != 0 and w2[41, dig] != 0 and w2[42, bal] != 0      # pass-42 readout pair
    assert w1[tot, 39] == w1[tot, 40] == -w1[ref, 39] == -w1[ref, 40]       # modulo-ten hinges
    assert w1[ctr, 43] != 0 and w2[43, lay["off"][0]] != 0                  # clock unit

    # 3. modulo hinges: total - digit (a zero-sum pair under equal LN2 gains) replaces total - reference
    assert W["h0.ln2.w"][dig] == W["h0.ln2.w"][tot]
    for u in (39, 40):
        w1[dig, u], w1[ref, u] = -w1[tot, u], 0
    # 4. counter offset folded into the clock unit's bias: 32*(r - 41) + 32*41 == 32*r
    b1[43] += w1[ctr, 43] * W["h0.ln2.b"][ctr]
    W["h0.ln2.b"][ctr] = 0
    # 1. delete the pass-42 readout units
    keep = [u for u in range(w1.shape[1]) if u not in (41, 42)]
    W["h0.mlp.w1"], W["h0.mlp.b1"], W["h0.mlp.w2"] = w1[:, keep].copy(), b1[keep].copy(), w2[keep].copy()
    # 2. continuous tied readout of total - digit
    for j in range(10):
        W["wte"][j, tot] = j
    W["wte"][:, bal] = 0
    g = W["lnf.w"][dig]
    assert g == W["lnf.w"][bal] == 2.0 ** 24
    W["lnf.w"][bal], W["lnf.w"][dig], W["lnf.w"][tot] = 0, -g, g
    W["lnf.b"][tot] = -1
    # 5. busy marker on '*': logit 4*13*(t - 1/4) - 4*71.5 - 5/4 = 52t - 300.25
    W["wte"][STAR, dig], W["wte"][STAR, tot], W["wte"][STAR, tri] = 13, 13, 71.5

    W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}
    for k, v in W.items():   # every weight stays exactly representable in bfloat16 (at most 8 significant bits)
        m, _ = np.frexp(v.astype(np.float64))
        assert np.array_equal(m * 2 ** 8, np.round(m * 2 ** 8)), k
    return W


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    W = build()
    (out / "config.json").write_text(json.dumps(CFG, indent=2))
    for name, value in W.items():
        np.save(out / (name + ".npy"), value)


if __name__ == "__main__":
    main()
