# Looped network, adaptive depth: correct for every K from 41 to 64

> Written by Claude Opus 5.5. The network is Astra's 364-nonzero looped network ([looped.md](looped.md)) with five
> closed-form edits by Opus, in [build/looped_adaptive.py](../build/looped_adaptive.py). That program runs Astra's
> `build/looped.py` unchanged and then applies the edits. No training, fitting or search was used.

## Result

- Config: `n_layer=1, loop=41, n_head=2, d_head=7, d_model=50, d_mlp=42`. Here `loop` is K_min: any K from 41 to 64
  gives the same strictly correct answers.
- **355 nonzero weights** (364 − 9). The dense size is 9,286. Every weight is exactly representable in bfloat16.
- Halting signal: an answer position whose column is still being reduced shows `*` as its argmax. The rule "run at
  least 6 passes, then stop at the first pass where no answer position shows `*`" stops at the first pass from which
  the answer is strictly correct.
- Measured results are in the [README](../README.md#adaptive-depth).

## Why the original network needs exactly 42 passes

Measured on 1,000 problems with fp32 weights, tracing the residual stream after every pass:

- **The column value is not the problem.** The value at coordinate 44 is final by pass 41 and bit-stable through
  pass 64; the subtract-10 hinges output exactly 0 below 9.125. The clocks `(r−4)²` and `(r−5)²` and the gathered
  column sums keep growing, but every gate they drive stays closed.
- **K < 42 fails because the readout has not run.** The answer is read from the digit coordinates 0/1. Until pass
  42 they still hold the input token: the visible previous answer digit.
- **K > 42 fails because the readout gate never closes.** Units 41/42 copy `total + 0.75` into the digit coordinate
  under the gate `255·(r − 41)`, which stays open after pass 42. In the packed build, unit 42 reads coordinate 0 but
  writes coordinate 1, so coordinate 1 grows by about `255·(r − 41)` every pass: 266 at K=43, 787 at K=44, 64,752
  at K=64. The readout reads coordinate 0 minus coordinate 1, so every digit is wrong from K=43 on.

## The edits

1. **Delete the readout units 41/42** (−11 nonzero weights, d_mlp 44 → 42).
2. **Read the answer continuously.** Each digit token j now also carries j on the total coordinate, so at an answer
   position the total coordinate holds `d + t`, where d is the visible input digit. The final LayerNorm gains
   become −2²⁴ on the digit coordinate and +2²⁴ on the total coordinate. The tied readout then gives digit j the
   logit `4j·((d + t) − d) = 4j·t`: the input digit cancels exactly. The pair is also mean-rejecting, because the
   gains times the embedding sum to zero for every token. Two more changes complete the readout:
   - `lnf.b[total] = −1` adds `−j`;
   - the existing triangle bias `−4·T(j)` is unchanged.

   This reproduces the old logit `4j(t + ¼) − 2j² + 5/4` exactly. The old balance column `−j` (9 entries) is
   removed. Net: +9 −9 +1.
3. **Make the subtract-10 hinges read the same pair.** They read `total − digit` instead of `total − (zero
   reference)`. This is the same count, and the reference entry becomes the digit entry.
4. **Fold the counter offset.** The counter's LN2 bias −41 existed only for the pass-42 gate. The clock unit read
   `32·(r − 41) + 1312`; it now reads `32·r` with bias 0. The pre-activation is the same and 2 nonzero weights go.
5. **Busy marker** (+3 nonzero weights). The `*` token gets the codes 13 (digit), 13 (total) and 71.5 (triangle).
   Its logit becomes `4·13·(t − ¼) − 4·71.5 − 5/4 = 52t − 300.25`, which overtakes digit 9's logit
   `36t − 151.75` at t = 9.28125. That threshold lies between the largest finished value (9.006, the column value
   9 with its small overestimate) and the smallest unfinished one (9.53, the value 10 with its largest
   underestimate of 0.47). Below the threshold, `*` stays under the winning digit: its slope 52 is steeper than every
   digit line, and it is already below the upper envelope of the digit lines at 9.006.

   The code 13 is bounded by the pass-5 routing gates. The `*` position holds digit code 13, so its multiplier
   products reach about 9·13 = 117. That is still under the gate offset of 128, so no product is routed from that
   position. A first try with code 18 gave products up to 162, which opened those gates and broke the arithmetic.

## Passes needed

After the column values are assembled on pass 6, the hinges subtract 10 once per pass while the value is ≥ 9.375.
Column k is therefore final after `6 + ⌊V_k/10⌋` passes, where V_k is its carry-corrected column value. The answer
is strictly correct from pass `6 + max_k ⌊V_k/10⌋` on, and from then on it no longer changes:
- the hinges output exactly 0 (the GELU tails saturate to 0 in fp32);
- the assembly and routing gates stay closed;
- the readout reads the frozen value.

The largest possible column value is 350: column 3 of 9999 × 9999, 4·81 + a carry of 26. So 41 passes suffice for
every operand pair, and 9999 × 9999 needs all 41. Random problems need far fewer; the distribution is in the
README.

The `*` marker appears exactly while some column still has t ≥ 9.375. So the halting rule "at least 6 passes, then
stop at the first pass with no `*`" stops at the first correct pass. Before pass 6 nothing has been assembled and all
positions show digit 0, which is why the rule waits for 6 passes.

## Nonzero accounting

| Tensor | looped (42 passes) | adaptive | change |
|---|---:|---:|---:|
| `wte` | 54 | 57 | +3 (digit codes moved from the balance column to total; `*` codes) |
| `lnf.b` | 2 | 3 | +1 (`−1` on total) |
| `lnf.w` | 2 | 2 | 0 (balance gain → total gain) |
| `h0.ln2.b` | 15 | 14 | −1 (counter offset) |
| `h0.mlp.w1` | 141 | 135 | −6 (readout units; the hinge reference becomes the digit entry) |
| `h0.mlp.b1` | 5 | 2 | −3 (readout units, clock unit bias) |
| `h0.mlp.w2` | 49 | 46 | −3 (readout units) |
| all others | 96 | 96 | 0 |
| **Total** | **364** | **355** | **−9** |

So the adaptive version is smaller. The one-shot overwrite cost 11 weights; reading the answer directly costs 1
(the readout's offset bias). The halting signal costs 3.

## Caveats

- **Measured, not proven.** The correctness argument bounds the column values; the float32 rounding of every
  intermediate is covered by the measured margins, not by a proof.
- **The halting rule includes a fixed preamble.** "At least 6 passes" is a property of this program: the column
  values are assembled on pass 6.
- **The extra passes are not entirely idle.** The clocks, the counter and the gathered sums keep growing after the
  answer is final. They only shift the final LayerNorm's statistics by a few parts in 10⁸, but the logits are not
  bitwise constant from pass to pass. The argmax and the strict correctness are.
