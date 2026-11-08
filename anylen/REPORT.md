# MULT-ANYLEN: one hand-programmed transformer that multiplies at every length

Editor: Claude Opus 5.5. `build.py` sets every weight in closed form. No gradients, training, optimizer, weight or mask
search, per-length weights or lookup tables indexed by length.

## Result

One weight set (`weights/anylen.npz`, sha256 `c9fdfea3a2c35c50...`, 870 nonzero of 139,440) multiplies an n-digit by an
m-digit number. It does this through a single looped two-block body and it decides itself when it has finished.

- **Measured.** All 3,053,242 problems were strictly right at the pass where the network halted, with 0 wrong in fp32 and
  bf16-rounded weights. That covers exhaustive n, m <= 3, at least 100k random per n for n = 4..20, at least 1k at n = 32,
  50, 100 and 200, unequal lengths, all-9s and powers of 10, and 33 problems at n = 300 and 500. Every run halted at
  exactly 2n + 3m + 2 passes, and answers stayed right on every later pass tested.
- **Float32.** No failure was found at any length tested, up to 500 x 500 digits. The strict margin stays at or above
  0.999981 from 1 x 1 digits to 500 x 500 digits and does not fall with length. So a measured float32 N_max is at least 500, and
  it is a lower bound: the run ended there, not a failure.
- **Exact arithmetic.** An argument for all n, m is in the section below. It is a hand argument with computed constants,
  not a machine-checked proof.

The network needs a start token, a looped body and an outer halting loop. Passes grow linearly with length, so this is
not a fixed-depth forward pass. The caveats are listed in full below.

**What kind of result this is.** The network's attention is position-only. Every q and k column and bias is zero in
`weights/anylen.npz`, so attention weights depend on distance alone and tokens never choose what they read. Each pass is a
read of earlier positions at fixed distances plus a per-position MLP. So the network runs as a one-dimensional cellular
automaton on a transformer block, with information moving left to right (in float32 the read is exactly 13 positions
wide; see Float32). This is an existence result in a well-known class and not a new capability. Iterative-array
multipliers date from Atrubin (invented 1962, published 1965), and a network trained by gradient descent reached
length-general binary multiplication in 2015 (see the comparison section).

## Network and format

`arch.py` is mult-min-net's `arch.py` with the following changes.

| Item | mult-min-net | This network |
|---|---|---|
| Input | `a * b EOS H1 H2 c` | `BOS a * b EOS c` (format change). BOS takes token 12, the old H1; H1 and H2 markers are gone |
| Position signal | `wpe` table of 20 positions | none. ALiBi-style bias `-slope * (t - s)` per head (Press et al. 2022) |
| Vocabulary | 14 | 14. Old H2 slot (13) is BUSY, an output-only token |
| Depth | fixed `loop` in config | same two blocks repeated until the network stops showing BUSY |
| Config | n_layer, n_head, d_head, d_model, d_mlp, loop | n_layer 2, n_head 2, d_head 40, d_model 128, d_mlp 104, alibi slopes |

Causal attention, GELU MLP, pre-LayerNorm and the tied output layer are unchanged. Input digits stay least-significant
first, and the answer is zero-padded to n + m digits.

BOS gives the operand's left end a fixed anchor. Without a position table, a run of identical tokens has no sharp way to
mark where it starts. The coordinator accepted this as a format change.

### Why the design is length-free

Pre-LayerNorm bounds every query and key, so content-addressed attention cannot stay sharp at unbounded length. The only
sharp reach is a distance bias, and that is local. So all long-range work has to be done by moving data, one position per
pass: an iterative array (Atrubin 1965), the structure later called a systolic array (Kung 1982). It is not Atrubin's
circuit. His array is real-time, emitting the nth product digit a fixed delay after the nth input digits. This one starts
with all digits in place and finishes in 2n + 3m + 2 passes.

- **a-stream.** Each pass, a position takes its left neighbour's S. BOS sits still and holds S = 0 with a tail flag, so
  behind a_0 the stream is flagged zeros.
- **b stays put.** A b position multiplies its own digit by the stream element passing it.
- **Partial-sum packets** move at half speed through a two-slot delay line: Z takes the left neighbour's Y, Y takes own Z.
  The packet in Y at position p on pass t has index k = 2p - t - (n+3). It meets a_i at b_j exactly when i + j = k, so it
  collects each product of column k once.
- **Carries.** The lower packet at a position (Z, index k-1) hands its pending carry to the upper one (Y, index k), which
  holds V = units + carry-in + product <= 9 + 9 + 81 = 99 and splits it into V mod 10 and a new carry <= 9.
- **Readout.** Head flags set while the stream's tail passes b_0 mark packets -1, -2, .... When packet -1 reaches EOS, a
  freeze wave starts at EOS and moves one position per pass, meeting packet k in answer cell k, where its digit is captured
  once into Ans.
- **Halting.** BUSY's logit is 64 - 128 Fz_prev. A cell shows BUSY until its digit has reached Ans.

Nothing reads n or m. Each operation looks at a fixed neighbourhood and every register is bounded by a constant.

## Exact-arithmetic argument (all n, m)

**Step 1, the integer schedule.** Assume registers hold exact integers.
1. Stream element at position p on pass t is a_(p-1-t). A b_j sits at p = n + 2 + j, so i + j = 2p - t - (n+3) = k. A
   packet advances one position every two passes, so k is constant along its path and it visits b_0 ... b_(m-1) once each.
2. Column k therefore receives exactly the products a_i b_j with i + j = k, plus the carry out of column k-1. That is
   schoolbook multiplication, and the bound V <= 99 keeps units and carry in 0..9.
3. The freeze wave meets packet k at answer cell k after all its products and carries have arrived, because packets leave
   b only after the stream tail passes. The capture is edge-gated and fires once, so later passes change nothing. That
   is the fixed point. Past the capture, Ans is rewritten from itself only.
4. The last cell freezes at pass 2n + 3m + 2 (read off the schedule, and measured in every run below).

`algo_sim.py` runs this schedule on integers (2,123 cases, 0 bad). Tests support step 1 but are not its proof.

**Step 2, the network computes that schedule with an error that does not grow.** Write delta for the largest distance of
any register from its ideal integer. Induct on passes: delta stays well under 3/8 at each pass boundary.
- **LayerNorm is linear.** The anchor pair +-2^20 gives sigma = 2^17 at d_model 128 with relative change near 1e-9 from
  the registers (which are bounded), and reads subtract the always-zero `ref` coordinate so the mean cancels.
- **Attention is position-only.** q and k columns are zero, so weights depend on distance alone. With q = 1/4096 and
  normalized shift-head weights w_d = q^d (1-q)/(1-q^(p+1)), the left neighbour's coefficient in
  pred = own + 4097 (o_shift - o_self) is (1 - q^2)/(1 - q^(p+1)), which is 1 at p = 1 and 1 - q^2 beyond. The rest sums to
  at most 9/4095 for digit registers. This bound depends on neither p nor n.
- **MLP snaps.** A step's ramp is v - 1/2 +- 1/8, so an input within delta of an integer sits 3/8 - delta from the ramp
  and lands on a flat part. With delta <= 0.04 the units see |x| >= 680, where gelu_new equals ReLU up to a tail below
  2^-1000. So every snapped register returns to its integer each pass and errors cannot build up.
- **Products.** The gated product is exact on integer inputs. Input noise e on S reaches P as about 10 e near a kink, so
  P errs by about 10 * 9/4095 = 0.022 plus a small GELU term at the kinks (measured total 0.0224). Unsnapped V inherits
  that, then is thresholded with tolerance 3/8.

Each per-pass error is bounded by a constant set by the fixed registers and weights, not by n, so delta stays near 0.022
and the step 1 schedule is what runs.

**Step 3, logits.** With Ans - DIG = c, digit logits are 2jc - j^2 = c^2 - (j-c)^2, so the correct digit beats any other
by at least 1. BUSY's logit is 64 before freezing, when the best digit logit is 0, and -64 after, below c^2 >= 0. Non-digit
tokens have SQ = 256 and sit far below. Float64 measurement: margin 0.999999999.

Weak points of this argument:
- It is hand-written with computed constants, not machine-checked.
- Bounding the capture timing relies on the head-flag and freeze-wave construction, which is argued here and tested
  below rather than derived in full detail.

## Float32

Measured: no failure up to n = m = 500. So I cannot report where float32 first fails. Reason to expect no length effect:

1. `arch.forward` computes t - s on integers before casting, so short distances stay exact at any T.
2. exp(-8.32 d) underflows to exactly 0 by d = 13 in float32, and slope 128 gives exactly 0 at d = 1. Each position's
   update depends only on a 13-position window, so positions past the first 13 run the same local rule at every length.
3. Registers are bounded and snapped, so nothing has room to grow with n or with the pass count.

Measured largest register distance from an integer (`results/verify-drift2-451532730.json`, 15 registers, every pass to halt):

| n = m | 3 | 10 | 30 | 100 | 200 |
|---|---|---|---|---|---|
| fp32 | 0.00781 | 0.01994 | 0.03977 | 0.03934 | 0.03997 |
| fp64 | 0.01941 | 0.02241 | 0.01966 | 0.02242 | 0.02242 |

The float64 value matches the estimate above. Float32 adds rounding up to 0.040, flat from n = 30 to 200, against a tolerance
of 3/8. The remaining limits are compute and memory: this dense implementation builds a T x T bias per head.

## Verification

Strict: the correct logit must be strictly above all 13 others at each of the n + m answer positions, ties wrong, at the
pass where the halting rule stops. All seeds below were drawn with `os.urandom` at launch, after the freeze (commit
`b228802cf7`). Weights and `arch.py` sha256 are recorded in every result file and match the committed files.

| Run | Seed | Jobs | Problems |
|---|---|---|---|
| main2 | 2325096199 | 644/644 | 3,052,357 |
| big2 (n = 300, 500) | 561135432 | 15/15 | 33 |
| top200 (n = 200 top-up) | 53163452 | 71/71 | 852 |
| drift2 (with float64) | 451532730 | 9/9 | 40 |

Full per-cell tables: `results/summary-main2.md`, `-big2.md`, `-top200.md`, `-drift2.md`.

Summary, fp32. Every wrong count is 0 in every cell.

| Cell | Problems | Passes run to | Halt pass | Min margin at halt | Min margin, halt to end |
|---|---|---|---|---|---|
| exhaustive, all 9 cells n, m <= 3, leading zeros included | 1,232,100 | 4T | 2n + 3m + 2 | 0.999981 | 0.999981 |
| random n = m = 4..20 | >= 100,000 each | 3T | 5n + 2 | 0.999981 | 0.999981 |
| random n = m = 32, 50, 100 | >= 1,000 each | 4T | 5n + 2 | 0.999981 | 0.999981 |
| random n = m = 200 | 216 + 852 (top200) | 4T / 2T + 1 | 1002 | 0.999981 | 0.999981 |
| unequal 7x3, 3x7, 20x1, 1x20, 13x2, 2x13, 17x9, 9x17, 5x1, 1x5 | >= 10,000 each | 4T | 2n + 3m + 2 | >= 0.999981 | >= 0.999981 |
| unequal 100x1, 1x100, 100x3, 3x100, 64x7, 7x64 | >= 1,000 each | 4T | 2n + 3m + 2 | >= 0.999981 | >= 0.999981 |
| unequal 200x1, 1x200, 150x40, 40x150 | 204 each | 4T | 2n + 3m + 2 | >= 0.999981 | >= 0.999981 |
| all-9s, powers of 10 and patterns, 139 (n, m) cells, n = 1..32, 40, 50, 64, 100, 200 | 9,490 | 4T | 2n + 3m + 2 | >= 0.999981 | >= 0.999981 |
| one fixed pass count for all sizes, 1x1 to 20x20 (6 cells) | 100 each | 3,212 | 2n + 3m + 2 | >= 0.999981 | >= 0.999981 |
| big2: n = m = 300 / 500 | 24 / 9 | 2T + 1 | 1502 / 2502 | 0.999981 | 0.999981 |

Across main2:
- **Strictly wrong at halt:** 0.
- **Halted at any pass other than 2n + 3m + 2, or never halted:** 0.
- **Fixed-point failures:** 0. That means no pass from halt to the end of the run had margin <= 0.
- **Fixed pass counts K = 2T, 2T+1, 3T, and 4T where run:** 0 wrong, min margin 0.999981.
- **Halting agreement:** the tracked halting logits equalled `arch.forward`'s own halting mode bit for bit in all 5,146
  checks.

**bf16.** Every weight is already a bfloat16 number, so `to_bf16` changes nothing. fp32 and bf16 records are identical in
all 644 jobs, apart from seconds. Arithmetic was fp32 in both. bf16 arithmetic was not tested.

**Replay.** `build.py` in fresh processes is byte-identical to the frozen weights (twice at freeze, once again at close).

**Greedy decoding** (`greedy_check.py`, `results/greedy-local.json`). Using the network's own digits instead of the
teacher's: 78 problems up to 10 x 10, 0 wrong, BUSY never read. Small test, run locally.

## Scope limits and caveats

- **Passes grow with length.** About 1.25 T. It is one set of weights, but not a fixed-depth forward pass. I have no
  impossibility proof for fixed depth, only the boundedness argument above.
- **Halting uses an outer loop.** BUSY is network output, but stopping is a rule outside the network, as in the adaptive
  looped nets. No pass count is supplied. `arch.forward` has a runaway guard, `K_max = 100000`, which would raise at about
  n = 20,000. It is harness code, not a network limit.
- **No end-of-answer token.** Past the last digit the network emits 0. The caller counts n + m digits.
- **Config constants.** Four ALiBi slopes live in the config, not the 870 count. mult-min-net's table was counted as weights.
- **Coverage cuts.** The first v2 run was stopped after 3.4 h and $10.58 because preemptions plus all-or-nothing writes
  could have lost everything. It produced no results. To fit the $25 cap in the relaunch:
  - n >= 150 cells got about 200 problems, down from 1,000. Leftover budget then brought n = m = 200 to 1,068, with the
    852 extra problems run to 2T + 1. 200x1, 1x200, 150x40 and 40x150 stayed at 204 each.
  - The 100k cells ran to 3T, not 4T.
  - n = 300 and 500 ran only to 2T + 1, with 24 and 9 problems.
- **Scope of the bar.** Random problems for n = 21..31 and most other lengths were not tested directly.
- **Input must be well-formed.** Behaviour on malformed sequences is not defined or tested. BUSY must not appear in input.

## Compared with trained models (cited, not rerun)

The closest prior work is the Neural GPU. Training reached length-general multiplication in 2015, in the same class of
local network whose depth grows with length.

- **Kaiser & Sutskever 2015** (arXiv 1511.08228). A convolutional gated recurrent network applies 3-wide local kernels for
  as many steps as the input is long. Trained by gradient descent on binary numbers up to 20 bits, it reports 100% fully
  correct outputs at 20, 25, 200 and 2000 bits. Technical limits stopped testing much above 2000. Usually only a few of
  the 729 grid-search runs reached 2000 bits. A width-1 version, a one-dimensional grid, also reached 2000 bits. The paper
  compares the model to a cellular automaton and cites Atrubin's algorithm as the reason to expect long multiplication.
  Its authors did not manage to learn long decimal multiplication.
- **Price, Zaremba & Sutskever 2016** (arXiv 1611.00736). Larger models and a base 2 -> 4 -> 10 curriculum learned decimal
  multiplication. A model under 1% error on random 100-digit problems still failed much more often on structured inputs:
  single digits padded with many zeros, products that need long carries, and 1111111111 x 1111111111.
- **Freivalds & Liepins 2017** (arXiv 1702.08727). With each decimal digit encoded in binary, a trained variant went from
  5-digit training numbers to 50-digit tests, with 2 of 5 models under 1% error.
- McLeish et al. 2024 (arXiv 2405.17399, Abacus Embeddings) report about 99% on 100-digit addition after training on 20
  digits, and multiplication only up to about 15 digits, mostly in-distribution.
- Bai et al. 2025 (arXiv 2510.00184) report standard transformers failing at 4 x 4 multiplication (below 1%), and their
  ICoT method reaching 100%.

Decimal is not the main difference, since trained Neural GPU variants have multiplied decimal numbers since 2016. These
differences remain:
- **Programmed, not trained.** This network says nothing about whether its design can be learned.
- **Failure profile.** Structured inputs are where Price et al. found trained models failing. Here no wrong answer
  appeared on all-9s, powers of 10 or patterned inputs, and the exact-arithmetic argument does not depend on digit
  pattern. Their exact inputs were not all run here, though: leading zeros were tested only at n, m <= 3.
- **Architecture.** Neural GPU kernels see both sides, and it runs a step count set by input length. This network reads
  only to the left and signals its own halt. Its attention is position-only, so it uses no content-based attention.

McLeish et al. and Bai et al. are learned models with different training and scoring. As I read them, neither reports
length-general multiplication. None of this is a head-to-head result. For Atrubin I read the abstract and later
citations, not the full paper. Kung is H. T. Kung, "Why Systolic Architectures?", IEEE Computer 15(1), 1982. Atrubin is
A. J. Atrubin, "A One-Dimensional Real-Time Iterative Multiplier", IEEE Trans. Electronic Computers EC-14(3), 1965.

## Cost

Modal CPU, $23.06 of the $25 cap, from Modal billing (`manylen-verify`, 2026-11-05 18:00 to 2026-11-06 03:00 UTC).

| Item | Cost |
|---|---:|
| First v2 run, stopped, no results | $10.58 |
| main2 (644 jobs) | $9.93 |
| top200 | $1.72 |
| big2 | $0.51 |
| drift2 | $0.11 |
| v1 partial runs and timing | $0.22 |

Results are copied here. Volume `manylen-results` keeps the per-job records.

Reproduce: `python3 -B build.py`, then
`python3 -B cloud.py verify weights/anylen.npz --plan main --tag T --seed 2325096199` (spends Modal money).
