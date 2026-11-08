# WSP-STATE-SIGNAL protocol

Written 2026-11-07 before any measurement in this lane. Read before writing it: the reports and code of WSP-GD-GAP, MULT-LOGDEPTH, MULT-ANYLEN and TEACH-WRITTEN. Those results are prior, so they shaped the hypotheses. Changes after this commit go at the end as dated amendments, each saying whether a scored result had been seen.

Everything is measurement. No weights are written and no editor is used.

## Why

Written multipliers are 100% correct at 53-60% confidence, and their weight gradients are large (3e8 on `two`, about 4,000 on `anylen`) and land mostly on weights that are zero in the program. The draft post says that gradients do not vanish out to 252 passes. Those numbers are gradients on weights, summed over all passes. They do not say how the answer depends on the state at pass t. Weight-noise results do not say that either. Three questions follow.

1. Does a correction step that removes small errors also remove small upstream changes, as Bengio, Simard and Frasconi (1994) argue for robust latching?
2. With the forward pass held fixed, does a different backward rule restore a damaged network where the standard gradient does not?
3. On CLRS quicksort, where does a trained student fail: reading the state, taking one step, or keeping the state usable over many steps?

## Experiment 1: dependence across passes, no training

### Networks

| Id | Network | Lengths n = m | Passes K | Copies |
|---|---|---|---:|---:|
| `anylen` | written, sha256 c9fdfea3a2c3..., 870 nonzero | 4, 8, 20, 50 | 5n + 2 | 1 |
| `ntt` | MULT-LOGDEPTH session A, sha256 4a88144218b8..., L = 2n | 2, 8, 32 | 55, 79, 103 | 1 |
| `anylen-rand` | same architecture and K, GPT-2 init as `gapb.py` | same | same | seeds 1, 2, 3 |
| `ntt-rand` | same architecture and K, same init rule | 2, 8 | same | seeds 1, 2 |
| `ngpu` | trained `ng-bin20-64-0.0005-rx1`, seeds 1, 2 (working) | binary 8, 20, 64 | tape width 2n + 1 | 2 |
| `ngpu-fail` | same, seed 3 (the run that failed on final weights) | same | same | 1 |
| `ngpu-rand` | same architecture, constructor init | same | same | seeds 1, 2, 3 |

`anylen` is a cellular automaton and is labelled so in every table. The Neural GPU is a convolutional cellular automaton. `ntt` is not local.

If n = 64 for `ntt` fits the budget after the other cells, it is added as a labelled extra. It is not needed for any read.

Weights load read-only and are checked by sha256. Neural GPU checkpoints come from an earlier training run (copies in `../inputs/ngpu/`).

Every state x_t is the full token state after pass t, with x_0 the embedding. All three families are autonomous after x_0: nothing is injected in later passes. d = K - t is distance to the end. Distance grid: 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192 and K, those that exist. Native precision is float32; `anylen` and `ntt` at n <= 8 are repeated in float64 for the backward measures. Everything runs on CPU VMs. Answers use the same K for perturbed and clean runs, so a broken network still ends.

Problems are fresh: seed 20261107 + cell index, 16 per cell (8 for `ntt`). Digits uniform; leading digit 1-9 for `anylen`, the distribution it was verified on. Clean strict accuracy is measured in each cell first. A written net with any wrong clean problem stops that cell for investigation.

### (a) Backward sensitivity

At each d, per problem:

- `g_ce(d)`: L2 norm of d(cross-entropy) / d(x_{K-d}), the same loss as earlier lanes.
- `g_m(d)`: the same for M (defined below), which isn't scaled by confidence.
- Split by coordinate class, per token: `dynamic` coordinates change on the clean run (any two passes differ by more than 1e-3, or 1e-3 of the state rms for the Neural GPU and random nets); `static` coordinates do not. This is needed because a held register reaches the readout through the residual path at every distance and could keep the norm up without carrying information. For `anylen` and `ntt` it is also split by named register.
- `sigma1(d)`: top singular value of d(answer logits at K) / d(x_{K-d}), by 16 power iterations on 4 problems per cell. Lengths n <= 20 for `anylen`, n <= 8 for `ntt`.
- Per-pass weight gradient: weights are copied per pass, so the weight gradient splits into a contribution from each pass. This gives the share of the squared norm from the last d passes. Effective gradient depth is the smallest d holding 99%.

The pass-by-pass backward step keeps one pass in memory at a time.

### (b) Forward effect of a finite perturbation

A perturbation is added to x_t, then the same network runs to K. Each trial records whether any answer digit changed, the final state distance, and r = final L2 distance / initial L2 distance. A smaller set records distance at each later pass.

Outcomes: `flip` is any answer digit changed from the clean answer. `erased` is no flip with r <= 0.01. `persists` is no flip with r > 0.01.

Directions:

- **Counterfactual.** Take a second problem that differs from the first in one input digit, with the same length. Let x'_t be its state at pass t and Delta = x'_t - x_t. Perturb to x_t + alpha Delta, with alpha in 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1. This is the direction that carries real information about the input. For `anylen` the second problem keeps the first problem's teacher-forced answer tokens, so Delta contains input information only; a smoke check confirms that this still gives the second answer at alpha = 1.
- **Isotropic.** Gaussian on every coordinate, L2 size set as a multiple of ||Delta|| (0.001 to 10 by half decades).
- **Gradient.** Plus and minus the unit vector of d(M) / d(x_t), same size grid. Undefined when the gradient is exactly zero; that is recorded.
- **Register, `anylen` and `ntt` only.** A named register at one token and at all tokens, sizes 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1, 2, 4 in register units. `anylen`: S, Zu, Yu, P, Ans, Lsep, Fz, ref, A+. `ntt`: r0, x0, z0, plane0, G, ptr, stage0, query, out0, the anchor; paired (+v, -v) and single-coordinate.

The register grid at n = 50 and `ntt` n = 32 may be thinned for cost before any result is read. Any thinning is recorded.

### The main statistic

Let y and y' be the two answers. Let M(x) be the sum over digits k with y_k != y'_k of logit_k[y'_k] - logit_k[y_k], with labels held fixed so that nothing switches at argmax. Then:

P(d) = |grad M(x_t) . Delta| / |M(x'_t) - M(x_t)|

This is the share of the true change in the answer that a first-order prediction at x_t sees. P near 1 means the gradient predicts the real dependence. P near 0 means the dependence is real and the gradient cannot see it. Medians over problems with nonzero Delta and nonzero change, with bootstrap 95% intervals (2,000 resamples, fixed seed).

alpha*(d) is the smallest grid alpha that changes the answer, a median over problems. Path flatness is the share of alpha intervals on which M moves by less than 1% of the slope of the straight line from M(x_t) to M(x'_t).

### Frozen reads

**Backward.** Read on `dynamic` coordinates and on all coordinates, separately.
- `vanishes`: g(d) <= 1e-3 g(0) at every grid d >= 8, or exactly zero.
- `flat`: g(d) within a factor of 10 of g(0) at every d.
- `grows`: g(d) >= 10 g(0) at some d.
- `other`: anything else.

**Tension.** For a net and length, at grid distances d >= 8 with nonzero Delta:
- `present` needs all of: median P(d) <= 0.01 at every such d; alpha = 1 gives y' on every problem; a counterfactual at alpha = 0.02 is `erased` on at least 95% of trials at d >= 8; and some alpha < 1 flips the answer on at least 95% of problems.
- `absent`: median P(d) >= 0.1 at every such d.
- `mixed`: anything else, reported as such.

The rule is applied unchanged to all nets. If a random control reads `present`, the statistic is not separating and the report says so.

**Depth of the weight gradient.** If effective gradient depth is at most 4 at every length on `anylen`, the flat norm in the earlier lanes comes from the last few passes, and the post's sentence that the gradient "stays that size out to 252 passes" needs changing.

Predictions, stated so that they can fail. `anylen` and `ntt` read `present`, because registers are snapped each pass. `dynamic` backward reads `vanishes`. All-coordinates backward may read `flat` or `other` through held registers. Effective depth is at most 4. The trained Neural GPU reads `mixed`. Random controls read `absent`. No prediction is made for the size of alpha*.

### What this cannot show

It tests these networks at these states. It says nothing about states reached during training from random, and it is not a theorem. A finite-difference check at one state is not the 1994 result, which is about hyperbolic attractors and asymptotics.

## Experiment 3: one-step versus free-running on CLRS (second in budget order)

### Models

All 30 scored `best.npz` from TEACH-WRITTEN (`hints` 10, `nohint` 5, `lat` 5, `lathints` 5, `latanneal` 5), recovered from the merge order in that report, plus the written teacher (`params.npz` rebuilt by `inputs/build_quicksort.py`, sha256 57a974a82d27...). Same code, clrs @6d5728b9, same `TNet`.

### Test sets

Official-style sets at n = 16, 64, 128 with 32, 32, 8 instances, seed 3, pos = i/n, as TEACH-WRITTEN scored them. Plus a fresh set of the same sizes, seed 20261107. No checkpoint was selected on either: selection was validation at n = 16.

**Replay gate.** Free-running F1 on the seed 3 sets must match each run's `result.json`. TEACH-WRITTEN's own replay matched 26 of 30 exactly and the rest within 0.0034. A wider gap stops the experiment for investigation.

### Modes

- `F`, free-running: the official evaluation. The model feeds back its own soft hints and carries its own hidden state.
- `R`, reference-fed: true hints at every step, own hidden carried. The decoder predicts the next step's hints, so each step is a one-step prediction from a true hint state.
- `Z`, reference-fed memoryless: true hints at every step, hidden set to zero at every step. This measures what hints alone give.
- `H_s`, reference-fed through step s, then free-running, s = 25%, 50%, 75% of each instance's length.
- `B_s`, free-running through step s, then reference-fed, same s. This tests whether the model recovers once the hints are right again.
- `T`, matched arms only (`lat`, `lathints`, `latanneal`): true hints plus the teacher's own hidden state in the student's normalized coordinates at every step. Only these arms share the teacher's coordinates. `T` is not run on `hints` or `nohint`.
- Teacher: `F`, `R`, `Z`, and `Rh` (true hints, teacher's own hidden). The teacher keeps array order partly in about e^-30 of soft mass in its fed-back hints, and reference hints are hard one-hots, so `R` may break it. That would be a finding about reference feeding.
- `nohint` has no hint encoders, so it gets `F` only, read by probes, as a floor.

Hints are scored as in TEACH-WRITTEN, strict per instance and step. `pred_h` is also scored per element. Own decoders score `hints` and `lathints`. Probes score every arm; they were fit on free-running latents at n <= 16, which is a limit on `R` readings.

### Decomposition

- **Reading the state.** Under `R`, accuracy where the target does not change between t and t+1, on p, r, i, j and on `pred_h` elements. The model only has to carry the value through. Also output F1 under `R`: at the last step the reference `pred_h` is the sorted order, so output F1 under `R` is a reading test and not an algorithm test. It must not be reported as the model running quicksort.
- **Taking one step.** Under `R`, accuracy where the target changes.
- **Keeping the state.** `R` minus `F`, by step and in the last quarter of the trajectory; `H_s` accuracy against horizon after the handoff, with half-life in steps; and whether `B_s` returns to the `R` curve.
- **Compounding check.** From `R`, the per-step chance c_t that all of p, r, i, j are right. Predicted free-running survival is the product of c_s up to t. Compare with actual survival under `F`. Actual much worse than predicted means errors feed each other or the state degrades; about equal means one-step error is the whole story.

### The recursion stack

Official hints have no stack. In CLRS `quicksort`, the hints are `pred_h`, p, r, i, j. When a partition's two children are both trivial, the next interval comes from pending recursion. Those pop steps are not a function of the current hints plus a local rule. The keys and the array do determine them in principle, because the whole run is a function of the input, so hints are not strictly non-Markov; but reaching the pending range means memory or re-running.

Measured from reference trajectories with an explicit simulator, per n:
- each transition is labelled `scan` (same interval), `child` (next interval follows from the current p, r and pivot), or `pop`;
- the share of steps and of interval changes that are `pop`;
- the share of `pop` steps solved by a stated local rule that uses only current hints and keys, a lower bound on what a memoryless reader could do;
- p and r accuracy on each event type under `F`, `R`, `Z` and `T`.

`R` accuracy above `Z` accuracy on `pop` steps would be direct evidence that a student keeps pending ranges in hidden state. Reference-fed evaluation therefore cannot show the model executing quicksort, since each step gets a state it did not produce and the stack is not supplied; it can show only where one-step prediction holds.

### Frozen reads

Means over seeds per arm and n, gate 0.9, computed separately on both test sets. All gates are reported; none stops the others.

- `reading` fails if `R` unchanged accuracy (mean of p, r, i, j) or `R` output F1 is below 0.9.
- `one step` fails if `R` changed accuracy (same mean) is below 0.9.
- `keeping` fails if `R` minus `F` on the last quarter of steps is at least 0.1.
- `stack`: the `R` versus `Z` difference on p, r at pop steps. Above 0.1 with a 95% interval over seeds excluding zero counts as a student holding pending ranges.

Prediction: reading holds at n = 16 and weakens with n. At n = 64 and 128 one-step accuracy is the first gate to fail, before keeping. No arm clears `pop`.

### What this cannot show

One algorithm, frozen checkpoints trained at n <= 16 on soft self-feeding, evaluated on hard reference hints they were not trained on. Reference-fed scores may be biased in either direction.

## Experiment 2: same forward computation, different backward rule (third in budget order)

### Networks and damage

`anylen` and `two` (487 nonzero, 4 x 4). `ntt` is left out for cost. Attention in `anylen` is position-only, so its GELU hinges are the only correction operations; `two` has real softmax reads, which the surrogate leaves alone.

| Cell | Damage | `anylen` | `two` |
|---|---|---:|---:|
| `n1` | relative Gaussian on nonzero weights | 1e-6 | 3e-8 |
| `n2` | same, larger | 1e-5 | 1e-6 |
| `z1` | zero one nonzero weight | 1 | 1 |
| `z4` | zero four nonzero weights | 4 | 4 |
| `s1` | scale one nonzero weight by 1.1 | 1 | 1 |
| `p` | positive control: relative Gaussian on `lnf` only | dev-chosen | dev-chosen |

A draw counts only if it is measurably wrong: strict accuracy below 0.95 on 256 dev problems at length 4. Draws are rejected in stream order with a fixed seed; the rejection rate is reported. For `p`, sigma is the smallest of 1e-3, 1e-2, 0.1, 0.3 that gives wrong draws. In `z1`, `z4` and `s1` the entries are uniform over nonzero weights outside `lnf`.

### Factors, 2 x 2 x 2

- **Loss.** `ce`: cross-entropy. `hinge`: mean of max(0, 0.5 - margin) per answer digit, margin being correct logit minus best other. A clean network has margin near 1, so the hinge is exactly zero there and pushes on nothing that is already right. This removes the push from 53-60% confidence.
- **Backward.** `std`: autograd. `sur`: the forward pass is bit-identical. In backward only, the derivative of gelu_new(u) at a unit is replaced by sigmoid(u / (k tau)), with k the unit's largest absolute input weight at the start of repair. For a snapped staircase this turns the flat steps into slope near 1, the straight-through estimator through snapping. tau is 0.125, 0.5 or 2, chosen on dev.
- **Support.** `all`: every entry may move. `nz`: only the program's nonzero support moves, with lesioned entries included. This is a diagnostic: a real learner doesn't know the support.

Identical forward pass is asserted in code: `sur` and `std` losses and logits must match exactly on a test batch.

### Optimizer and selection

Relative Adam: delta w = -lr x c x adam_direction, with c the rms of the tensor's nonzero entries at the start of repair (the median of those values for an all-zero tensor), because weights span 1/64 to 2^18. lr in 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2. 1,500 steps, batch 64, lengths 1 to 4 for `anylen`, 4 x 4 for `two`. Non-finite gradients skip the step and are counted.

Every 100 steps a monitor scores 256 problems at length 4, and 64 at each length pair for `anylen`. The checkpoint kept is the first with monitor strict 1.0, else the best monitor. This lets `ce` pass through a correct point and be credited, then reports separately whether it left.

lr and tau are chosen per net, cell and arm on 2 dev draws, by restored share, then final accuracy, then smaller lr. Scored: 12 new draws per cell and net, with problems from a separate seed. All arms get the same draws.

**Restored** means strict 1.0 on fresh problems: 4,096 at length 4, 256 at 20, 32 at 100 for `anylen`; 4,096 for `two`. Reported with final accuracy and relative distance to the clean weights, both on the support and overall, since function can come back on different weights.

### Diagnostics at the damaged point

No optimizer involved, so no lr choice affects them. Per draw, for each loss and for `std` and `sur` at tau 0.5:
- cosine between the update direction and (clean - damaged), over all entries and over the support;
- share of squared gradient on zero entries;
- in `z1`, the rank of the lesioned entry among all entries by gradient magnitude.

Line check: strict accuracy and loss along damaged + beta (clean - damaged), beta 0, 0.1, ... 1. If accuracy only returns at beta near 1, the repair target is a needle even along the right line.

### Frozen reads

- **Control.** `p` must be restored by `std` on at least 50% of draws in some arm. If not, the pipeline is suspect and a null below cannot be read.
- **Surrogate helps.** In a cell, for fixed loss and support, restored share `sur` minus `std` at least 0.25 and exact McNemar p below 0.05 after Holm over all cells, nets, losses and supports in that comparison. This says the forward pass and the loss allow repair and the signal was the obstacle there.
- **Null.** No cell passes. Then the report says which cells had neither restoring and which had both. Neither restoring does not by itself show a forward problem; the line check and cosines say more.
- **Loss and support.** The same test for `hinge` versus `ce` and `nz` versus `all`, so a surrogate gain is not confused with a loss gain.

Prediction: surrogate helps in at least one lesion cell with `hinge`. Noise cells stay unrestored in every arm, because earlier lanes put the radius near one ulp, so repair would have to be exact on every weight.

### What this cannot show

Repair from a nearby point is not discovery from random. A null covers one surrogate family, one optimizer and this step count. `nz` knows the program's support, which no learner has.

## Standards

- Every headline number is recomputed from raw output by `verify.py`, which a reviewer can run. It includes a small independent recomputation that fits the local limit.
- Tables in `TABLES.md` are generated from raw JSON.
- Prior art is searched before anything is called interesting: Bengio, Simard and Frasconi 1994; Hochreiter 1991; Pascanu et al. 2013; straight-through estimators (Hinton 2012, Bengio et al. 2013, Yin et al. 2019); surrogate gradients (Neftci et al. 2019); teacher forcing and scheduled sampling (Williams and Zipser 1989, Bengio et al. 2015, Lamb et al. 2016); DAgger; recursive and hint-free algorithmic reasoning; Neural GPU follow-ups.
- The spurious line in the MULT-LOGDEPTH report, that the Neural GPU collapse came as saturation pull rose, is not reused: `pull` applies only at rx > 1, and seed 3 is rx1.

## Compute and budget

Ceiling $25 inclusive, planning figures from `resources.json`. Up to about $5 over is acceptable so that budget doesn't stop results, not for new scope.

| Line | Planned |
|---|---:|
| Experiment 1, CPU | $6 |
| Experiment 3, L4 Spot | $7 |
| Experiment 2, CPU | $6 |
| Reserve and verification | $6 |

CPU work uses CPU-only VMs, Experiment 3 uses L4 Spot as TEACH-WRITTEN did. Local compute stays within 60 s on one thread. `cloud.py` refuses any VM whose full lease would pass the ceiling. Every job has a coded timeout, each VM has a provider lease, and the agent uploads partial output every 2 minutes. Every case writes its own JSON when it finishes, a restarted job skips finished cases, and a rate check is done on the first finished cases before the rest are left to run.

If budget tightens the order is 1, 3, 2. If an experiment is blocked or too expensive the other two finish and the report says so.

## Amendments

### 2026-11-07 ~07:30 UTC, from the coordinating session

Status when written. Experiment 1: partial backward output had been read (`anylen`, `ntt`, random controls), which is why nothing about Experiment 1's reads changes here. Experiment 3: a smoke run at n = 16 on four checkpoints had been read. Experiment 2: only draw selection and one 100-step timing run per net had been read, both unrestored at step 100. No Experiment 2 dev, diagnostic or scored result had been read. So the first two additions below are pre-registered for Experiment 2.

1. **Denominators.** Beside every share of squared gradient on zero entries, the Experiment 2 diagnostics report rms gradient per coordinate on zero entries and on support entries, with the coordinate counts. `anylen` has 139,440 parameters, 99.38% zero; `two` has 231,658, 99.79% zero. The gradient is not to be described as pointing away from the program on `anylen` without those numbers.
2. **Output-gain control, no training.** Multiply the final answer logits by an external alpha > 0, a diagnostic outside the architecture. Every argmax is preserved. Report loss against alpha on `anylen` and `two`, and the alpha that gives mean loss 0.01. Report it beside the earlier fact that `lnf.w` x 1.1 breaks both nets, and find out why that scaling is not a pure logit gain. This separates "the objective asks for confidence" from "the architecture has no way to supply it". Script: `gain.py`.
3. **Prior art.** ALTA (arXiv 2410.18077) is compared where relevant, especially in Experiment 3.

### Implementation notes recorded at the same time, none changing a read

- Register grids at `anylen` n = 50 and `ntt` n = 32 were thinned for cost before any result: distances 1, 4, 16, 64, 192 and K; sizes 0.05, 0.2, 0.4, 0.6, 1.
- `ngpu` copies also get `sigma` up to n = 20, which the protocol left unstated.
- M uses the true digits y and y' for every net, including random ones, so it stays defined when a random net's argmax does not move. `flip` is against the net's own clean argmax.
- In Experiment 2, Adam's eps is 1e-8. `restored` checks lengths 20 and 100 only when the shorter length passed, since all must pass. The monitor scores the other length pairs only when length 4 is strict 1.0. A `hinge` run stops at its first monitor pass, because its gradient is then about zero; `ce` runs continue, so that leaving a correct point is seen.
- Experiment 2's cost estimate was about $5, inside its line.

### 2026-11-07 ~08:40 UTC, Experiment 2 grid cut for cost

Read before this was written: the rate only (91 of 2,304 dev runs per hour, so over 20 hours and about $20). No dev result was read, and those 91 runs are discarded unread. The diagnostics at the damaged point had been read (protocol amendment above), including cosines near 1e-9 on most cells, which is why this note says so.

Changes: 500 steps, not 1,500. Batch 32, not 64. lr 1e-7 to 1e-3, dropping 1e-2. tau 0.5 and 2, dropping 0.125. One dev draw per cell, not two. Scored draws stay at 12 and all 8 arms stay. A null therefore covers 500 steps at these rates. It says less about slower repair than the protocol would have said.

### 2026-11-07 ~09:00 UTC, control failed on dev; `dmg` support added

Read before this was written: the dev runs on `two` (not `anylen`), and no scored result. On `two`, every arm in every cell, including the positive control `p`, leaves strict accuracy at or below its start: even at the smallest lr, 1e-7, Adam's equal step on every allowed entry drives `two` to 0 by step 100. So the frozen control fails on dev for `two`, by the update rule rather than the signal.

The 8 frozen arms still run as frozen, and their reads stay as written, including that a null is not readable if the control fails.

Added, post hoc and labelled so in the report: a support level `dmg`, where only the entries the damage touched may move (`lnf` nonzero entries in `p`; the lesioned or scaled entries in `z1`, `z4`, `s1`; noise cells skip it because it equals `nz` there). It is an oracle. Four arms (loss x backward rule) on those four cells, lr 1e-7 to 1e-2, tau 0.5 and 2, same dev and scored draws, same selection rule. Purpose: with the other tight weights held still, does the gradient at the damaged entries repair them, and does the surrogate differ? Its `sur` against `std` tests are uncorrected and exploratory.

### 2026-11-07 ~09:20 UTC, Experiment 4 added: does a scaled network survive training?

Requested by the user through the coordinating session ("run this"). Written before any Experiment 4 code ran. Already known and shaping it: `gain2.py` (scaling `lnf.w` and `lnf.b` together keeps 100% at CE about 0.01; `anylen`'s gradient falls about 11x, `two`'s under 2x), and Experiment 1's read.

**Question.** Is training's destruction of these nets about confidence (the loss still wants something) or about sensitivity (any small move along the gradient breaks them)?

**Conditions.** `base`: clean weights. `scaled`: `lnf.w` and `lnf.b` times alpha, 5.2 on `anylen`, 6.72 on `two`, mean CE about 0.01. Same batches, same draws, same protocols.

**Matched step scan (primary, no training).** Removes the gradient's size and asks only about its direction. For each of 8 gradient batches (n = m = 4, 64 problems, fresh seeds), from each condition, step a fixed L2 weight-space distance eps along a unit direction and test strict accuracy. eps* is the largest distance with strict 1.0 before the first failure: half-decade grid 1e-8 to 1e6, then 8 log-bisection steps. Failure is any wrong problem at length 4 (256 problems) or, for `anylen`, 20 (32) or 100 (4); longer lengths run only when shorter ones pass. Directions: `grad` minus the unit CE gradient on all entries; `grad_noemb` the same with `wte`, `wpe` frozen; `grad_nz` the same on the support; `gain` plus the unit vector of (`lnf.w`, `lnf.b`) and `gain_neg` its negative, the known safe control; `rand` Gaussian on all entries; `rand_nz` Gaussian on the support. Also recorded: the cosine between the CE gradient and `gain`, CE along each line, and the lowest CE reached before failure, which answers whether CE at the scaled point can gain anything further without breaking. fp32, native.

**SGD.** Step size follows the gradient. All weights, and `wte`, `wpe` frozen. lr by decades: 1e-12 to 1e-4 on `anylen`, 1e-17 to 1e-9 on `two`, the same absolute lr in both conditions. 3 seeds, 2,000 steps, batch 64, lengths 1 to 4 for `anylen`, 4 x 4 for `two`. Strict at length 4 every 25 steps; lengths 20 and 100 every 100 steps. A run stops at first failure. Recorded: steps to first failure, displacement D from the start at the last passing check, CE at that check.

**Adam.** lr 1e-10 to 1e-3 by decades, same scopes and seeds, plain Adam. Adam normalizes gradient scale, so a lower loss does not shrink the step, and a scaled net could die for that alone. Reported separately with that caveat. It takes no part in the read.

**Frozen read, per net.**
- R_eps: median over gradient batches of eps*(scaled) / eps*(base) along `grad`, with a bootstrap 95% interval.
- R_D: median of D(scaled) / D(base) under SGD, paired by lr, scope and seed, over runs where both conditions failed before the cap, with a bootstrap interval. Runs where only one failed are counted and reported.
- `confidence implicated`: the lower bound of R_eps or of R_D is at least 3.
- `sensitivity`: both intervals lie inside [1/3, 3].
- `mixed`: anything else.

If `gain` does not survive far past `grad` in both conditions, the scan isn't separating and the report says so. Steps to failure alone isn't the read, because a smaller gradient under SGD gives a smaller step and so survives longer for that reason alone; steps are reported beside D.

**What this cannot show.** It starts from a correct network. It says nothing about reaching one from random. `scaled` changes only output gain, and the read is about these two nets at these points.

### 2026-11-07 ~12:00 UTC, post hoc lr sweep on the one cell where the surrogate differs

Read before this was written: all scored Experiment 2 results. In the oracle `dmg` arm on `anylen` `z1`, `sur` restored 5 of 12 and `std` 0 of 12, but `std` ran at its dev-chosen lr 1e-7 because it never restored on dev. To check that this isn't an lr artefact, both rules run at every lr (1e-7 to 1e-2) on the same 12 scored draws, 500 steps, both losses. Post hoc and labelled so. It does not change any frozen read.
