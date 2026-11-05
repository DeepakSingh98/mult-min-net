> Astra's (GPT-6 Astra, `codex exec`, effort high) own write-up of its adaptive edit, unchanged except this note. Its
> test numbers are from seeds it chose in its sandbox; the independent 1M fresh-seed and 9000..9999 results are in the
> README. "/work/build.py" is [build/looped_adaptive_astra.py](../build/looped_adaptive_astra.py) here, and
> "/work/prev/build/looped.py" is [build/looped.py](../build/looped.py). "Parameters" means nonzero weights.

# Adaptive hand-built multiplication transformer

## Delivered model
Config: {"n_layer": 1, "loop": 42, "n_head": 2, "d_head": 7, "d_model": 50, "d_mlp": 44}

**K_min=42; 365 nonzero parameters, +1 (+0.275%) versus 364.**
Dense count: 9,386. One shared block, unchanged forward computation.
The extra entry makes the readout balanced on every pass rather than one-shot.
Run `python /work/build.py OUTDIR` to generate config.json and all 16 full
float32 tensors. Fresh-process replay matched the installed config and every
tensor bit for bit. The builder never reads /work/edited.

## Attribution and method
The arithmetic core is the supplied earlier builder /work/prev/build/looped.py.
Its explanation documents analytic GELU multiplication, routing factors, carry
approximation and clocks; those are retained. This builder imports that allowed
program and changes exactly six scalar entries, one of which was zero.
No training, autograd, optimizer, automated weight/mask search, regression, or
data fitting was used. All changes follow from the readout recurrence.
Diagnostic data were used to test the construction, not calculate weights.

## Measured cause of the original failure
Baseline seed123, N128: strictly correct 128/128 at K42 and 0/128 at
K40,41,43,44,50,64. Before K42 the readout gate is closed; the original input
digit code remains even when arithmetic is done.
At K42 the readout approximately sets digit d=x+0.75 and balance b=1.
Later it continues overwriting d, but b accumulates d+255*(r-41)+1 without
a cancelling term. In the first cached problem remainders were unchanged
from K40 through K64, while b was about 1 at K42, 258–265 at K43, and
64,552–64,708 at K64. Thus post-K42 failure was measured in the readout,
not assumed to come from the arithmetic. One-shot routing clocks are retained.

## Exact edits and recurrence
- h0.mlp.w2[41,1]: 0 -> -1/32.
- h0.mlp.b1[41]: 56 -> 24 (additive term 1.75 -> 0.75).
- lnf.w[0] and [1]: 2^24 -> 2^23.
- h0.ln2.b[24]: -41 -> -5.
- h0.mlp.b1[43]: 1312 -> 160.

The last two edits open readout at pass 6 while preserving the absolute clock
update argument 32*((r-5)+5)=32*r.
Initial embeddings satisfy b=-d. Both readout units now write opposite updates
to d and b, preserving b=-d. When active, in the linear GELU regime:

    U = x + 255*(r-5) + 0.75
    V = d + 255*(r-5) + 1
    d_new = d + U - V = x - 0.25
    b_new = -d_new

The gate is inactive during passes 1–5. Routing therefore consumes unchanged
input digits at pass 5; pass 6 gathers and assembles these fixed columns.
Only then does the readout change input digit coordinates. Later multiplier
and gather activity is ignored by closed routing/assembly gates.
Readout observes the pre-MLP total, one pass behind modulo updates.
A 28-problem cached audit (24 random, four edge cases) measured maximum
|d+b|=0.0, and maximum absolute GELU activation of routing/assembly
units 8–38 after pass 6 equal to 0.0.

## Stable answer
The inherited arithmetic constructs column integers V<=350 with a small
approximation error. The modulo units implement

    x_new = x - (40/64) * [g(64*(x-9.125)) - g(64*(x-9.375))].

This subtracts ten above the transition and zero below it. The inherited carry
approximation underestimates by at most about 0.5 before floating-point errors,
separating representations of integer 9 and integer 10 across the transition.
Thirty-five subtraction passes, 7–41, suffice; K42 reads the finished remainder.

With balanced code and halved final gains, output digit j has approximate logit

    L_j = 4*j*(x-0.25) - 4*T(j) + 1.25, where T(j)=j*(j-1)/2.

Markers have logit -1.25. The recurrent overwrite preserves the winning digit
once the remainder is finished, rather than accumulating a counter-dependent
balance. This is an ANSWER fixed point, not bitwise residual/logit convergence:
clocks and unused gathers continue and cancellation introduces small jitter.
The claim is scoped to K<=64, not arbitrarily many passes.

9999*9999 is wrong at K40 and K41 (fourth digit predicted 9, not 0) and correct
at K42 through K64. Its maximum logit1-logit0 is 38.1665 at K41 and 33.6495
at K42. Thus K_min=42 is supported by a worst-case example, not chosen from
the smaller maximum seen in random sampling.

## Output-only halting rule
For the full teacher-forced sequence, after each pass K>=7, compute
S_K=max over positions 11..18 of (logit[token1]-logit[token0]).
**Stop at the first K>=7 with S_K<36**, then return all eight argmax digits.
Only outputs and pass count are inspected. No targets or hidden state.
If no threshold crossing occurs by 64, report failure, not an uncertified answer.
K6 is excluded because it reads the zero pre-assembly total.

The logit difference is approximately 4*(x-0.25), so 36 corresponds to x=9.25,
midway between the subtraction thresholds. It avoids the unfinished argmax-9
plateau. Naive unchanged-all-eight-argmax for one pass, first eligible at K7,
stopped wrongly on 93,011/100,000 random tests; do NOT use that rule.

For actual greedy generation, use the same threshold at the current prediction
position from K7, emit that digit, and restart the ordinary forward computation
with the extended prefix. No true answer tokens are supplied. Testing 1,000
random problems (seed8675309) plus four edge cases gave 1,004/1,004 correct,
zero missing halts, and minimum winner margin 0.985504. Individual digits used
7–42 passes. Eight fresh-prefix runs averaged 90.42231 total block passes
versus 336 for eight fixed-42 runs. This is a separate pass-count measurement,
not a wall-clock benchmark or a resumable-cache claim.

## Exact-trajectory measurements
Instrumentation matched official forward bit for bit at K6,7,15,42,64.
An initial diagnostic used different MLP bias-addition grouping; it was corrected
and all reported halting measurements below were rerun with exact grouping.

100,000 random problems, seed20261105:
- Zero wrong stops, zero later strict failures through K64, all stopped.
- Mean halt **16.45888**, median16, p90=22, p95=24, p99=27, p99.9=31.
- Range7–36; mean savings25.54112 passes, or60.8122% versus42.
- Oracle first permanently strictly-correct pass: mean16.309, range7–36.
  This oracle is retrospective only, not used for halting.
- Every wrong-answer state at K>=7 had S_K>=37.015625.
  Already-correct states may exceed the threshold: halting is conservative.
- Minimum strict margin K42..64:0.97088623046875.

28,256 boundary entries, duplicates intentionally retained:
all a=1000..9999 with b=9999, their reversals, Cartesian square9900..9999,
and a16x16 Cartesian corner set with leading digit1 or9 and lower digits0 or9.
- Zero wrong stops or later failures; all stopped; range7–42.
- Mean halt25.2595201.
- Every wrong-answer state at K>=7 had S_K>=37.011474609375.
- Minimum strict margin K42..64:0.974151611328125.

Full random distribution:
| Pass | Halting count | First permanently correct count |
|---:|---:|---:|
| 7 | 72 | 77 |
| 8 | 519 | 589 |
| 9 | 1452 | 1561 |
| 10 | 2822 | 3109 |
| 11 | 4648 | 4956 |
| 12 | 6449 | 6755 |
| 13 | 8206 | 8494 |
| 14 | 9666 | 9902 |
| 15 | 10324 | 10347 |
| 16 | 10106 | 10062 |
| 17 | 9123 | 8975 |
| 18 | 8173 | 7931 |
| 19 | 7008 | 6762 |
| 20 | 5712 | 5491 |
| 21 | 4578 | 4409 |
| 22 | 3364 | 3222 |
| 23 | 2467 | 2367 |
| 24 | 1838 | 1734 |
| 25 | 1272 | 1190 |
| 26 | 831 | 794 |
| 27 | 540 | 506 |
| 28 | 351 | 324 |
| 29 | 213 | 188 |
| 30 | 124 | 118 |
| 31 | 58 | 55 |
| 32 | 38 | 36 |
| 33 | 24 | 24 |
| 34 | 15 | 15 |
| 35 | 5 | 5 |
| 36 | 2 | 2 |

All savings above are block-pass counts on specified workloads, not wall-clock speedups.

## Required acceptance and bfloat16
Official check(n=100000,seed=7): PASS.
Zero wrong at EVERY K42..64 in BOTH float32 and bf16-rounded weight modes.
Minimum margin in each mode:0.973358154296875.
Every delivered weight is already exactly bf16-representable, verified
entry by entry; rounding is identity. Both modes therefore execute identical
float32 arithmetic, including halting. No claim is made about bf16 computation.
New constants -1/32,24,2^23,-5,160 are all exactly representable.

## Nonzero table
The tied embedding is counted once; zero tensors are included.
| Tensor | Previous | Delivered | Change |
|---|---:|---:|---:|
| wte | 54 | 54 | +0 |
| wpe | 19 | 19 | +0 |
| lnf.w | 2 | 2 | +0 |
| lnf.b | 2 | 2 | +0 |
| h0.ln1.w | 12 | 12 | +0 |
| h0.ln1.b | 0 | 0 | +0 |
| h0.attn.wqkv | 22 | 22 | +0 |
| h0.attn.bqkv | 2 | 2 | +0 |
| h0.attn.wo | 8 | 8 | +0 |
| h0.attn.bo | 0 | 0 | +0 |
| h0.ln2.w | 30 | 30 | +0 |
| h0.ln2.b | 15 | 15 | +0 |
| h0.mlp.w1 | 141 | 141 | +0 |
| h0.mlp.b1 | 5 | 5 | +0 |
| h0.mlp.w2 | 49 | 50 | +1 |
| h0.mlp.b2 | 3 | 3 | +0 |
| Total | 364 | 365 | +1 |

## Limits
No exhaustive all-81-million-pair floating-point proof is claimed. The analytical
construction, strict margins, random tests and boundary sweep support the result.
The experimenters' fresh million-problem check remains pending.
The least certain aspect is a formal bound on accumulated floating-point errors,
especially large-counter readout cancellation and LayerNorm scale changes.
No residual convergence or stability beyond64 is claimed.
365 nonzeros is not claimed parameter-minimal.

## Final independent unchanged-model acceptance
check(n=100000,seed=271828):

{
  "K_min": 42,
  "nonzero": 365,
  "fp32_wrong_by_K": {
    "42": 0,
    "43": 0,
    "44": 0,
    "45": 0,
    "46": 0,
    "47": 0,
    "48": 0,
    "49": 0,
    "50": 0,
    "51": 0,
    "52": 0,
    "53": 0,
    "54": 0,
    "55": 0,
    "56": 0,
    "57": 0,
    "58": 0,
    "59": 0,
    "60": 0,
    "61": 0,
    "62": 0,
    "63": 0,
    "64": 0
  },
  "fp32_min_margin": 0.972869873046875,
  "fp32_wrong": 0,
  "bf16_wrong_by_K": {
    "42": 0,
    "43": 0,
    "44": 0,
    "45": 0,
    "46": 0,
    "47": 0,
    "48": 0,
    "49": 0,
    "50": 0,
    "51": 0,
    "52": 0,
    "53": 0,
    "54": 0,
    "55": 0,
    "56": 0,
    "57": 0,
    "58": 0,
    "59": 0,
    "60": 0,
    "61": 0,
    "62": 0,
    "63": 0,
    "64": 0
  },
  "bf16_min_margin": 0.972869873046875,
  "bf16_wrong": 0,
  "pass": true
}
