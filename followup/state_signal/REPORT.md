# WSP-STATE-SIGNAL: why written networks and training don't meet

Lane run 2026-11-07, about 06:40 to 12:30 UTC, on GCP CPU and L4 Spot VMs. Measurement only: no weights were written and no editor was used. The protocol, five dated amendments and the decisions log are [PROTOCOL.md](PROTOCOL.md) and [DECISIONS.md](DECISIONS.md). Every table is generated from raw JSON in `results/` by `make_tables.py` into [TABLES.md](TABLES.md). `python3 -B verify.py` recomputes the headline numbers and runs small independent checks from the weights.

`anylen` and the Neural GPU are cellular automata: `anylen` is a one-dimensional automaton on transformer blocks, the Neural GPU a convolutional one. The NTT network (`ntt`) is not local.

## Short version

1. **The 1994 tension is there in the written networks, and in the trained Neural GPU too, but only in part.** A real upstream change, such as one input digit, decides the answer at every distance. The gradient at the state sees almost none of it: at least 8 passes from the end, the first-order prediction recovers at most 1e-28 of the true change on `anylen`, at most 2.5e-8 on `ntt`, and at most 1.5e-3 on the trained Neural GPU. Random-init controls and the Neural GPU seed that failed recover most of it, 0.5 to 1.3. Small changes do not reach the answer. Small changes are not erased from the state, though: held input copies on `anylen`, and registers no later pass reads on `ntt`, keep them. So the frozen read is `mixed`, not `present`.
2. **The weight gradient that earlier lanes reported as flat to 252 passes comes from the last pass alone.** d(loss)/d(state) falls by about 8 orders of magnitude in one pass on `anylen` and about 5 on `ntt` (about 7 on `ntt` by d = 12). Once the last pass and the readout are counted, the rest of the weight gradient is 1e-13 to 1e-8 of its squared norm. The draft post's sentence "It's wrong" about vanishing gradients is itself wrong for the state. The gradient does vanish across passes. It is large only at the last pass.
3. **With the forward pass fixed, no backward rule repaired any damage when all weights or all program weights could move.** On both nets, 0 of 1,152 scored runs were restored. Adam's equal step on every allowed entry breaks the other tightly tuned weights at once, and that also failed the positive control, so the frozen null can't be read as a property of the gradient signal. A post hoc arm that lets only the damaged entries move restores the `two` control 12 of 12 and some `anylen` damage. That arm has one real backward-rule effect. On a single zeroed `anylen` weight, the standard gradient restores 0 of 12 at every lr. The surrogate, which keeps the forward pass and passes gradient through the snapping plateaus, restores 5 of 12 (6 of 12 at 1,500 steps). This is post hoc and uses an oracle support.
4. **CLRS quicksort: reference-fed hints do not rescue the trained students, at any size.** One-step accuracy from the true hints is 0.59, 0.26 and 0.15 at n = 16, 64 and 128 for the official-hints arm, so taking one step already fails. Reading and keeping fail too. Hard reference hints lower output F1 below free-running, because the models never trained on them. Even the written teacher takes only 12-14% of its steps right from true hints, against 81-86% with its own hidden state added. So the official hints are not the state these networks compute in.
5. **A scaled network doesn't survive training any better (Experiment 4, added at the user's request).** Scaling `lnf.w` and `lnf.b` together is an exact logit gain inside the architecture. It takes CE to 0.01 at 100% accuracy. Along the CE gradient, scaled and unscaled nets break at the same distance: ratio 0.996 [0.935, 1.08] on `anylen` and 1.00 on `two`. The gain direction never broke up to 1e6. The cosine between the CE gradient and that direction is about 1e-10 or less. The frozen read is `mixed`, for the reasons in Experiment 4. What the measurements show is sensitivity, not confidence.

## Experiment 1: dependence across passes

Networks: `anylen` at n = 4, 8, 20, 50 (K = 22 to 252); `ntt` at n = 2, 8, 32 (K = 55, 79, 103); trained Neural GPU `ng-bin20-64-0.0005-rx1` seeds 1 and 2 (working) and 3 (failed), binary n = 8, 20, 64; random-init controls of each architecture. Fresh problems: 16 per cell, 8 for `ntt`. Clean strict accuracy 1.000 on every written and working-trained cell.

### Backward

Median L2 norm of d(CE)/d(state) at distance d from the end, float32. The full grid is in TABLES.md.

| net | n | d = 0 | d = 1 | d = 8 | d = K | class |
|---|---:|---:|---:|---:|---:|---|
| `anylen` | 4 | 0.15 | 2.1e-9 | 1.1e-9 | 1.1e-9 | vanishes |
| `anylen` | 50 | 0.037 | 8.6e-10 | 4.3e-10 | 4.3e-10 | vanishes |
| `ntt` | 2 | 0.45 | 2.3e-6 | 2.3e-6 | 2.9e-8 | vanishes |
| `ntt` | 32 | 0.11 | 5.8e-7 | 5.8e-7 | 7.9e-9 | vanishes |
| Neural GPU s1/s2 | 64 | 0.012 | 4.0e-4 | 6.2e-5 | 4.5e-5 | other |
| Neural GPU, failed seed 3 | 64 | 0.13 | 0.033 | 0.014 | 0.11 | other |
| random `ntt` | 8 | 0.0024 | 0.0024 | 0.0025 | 2.8 | grows |

- The float32 floor of about 1e-9 on `anylen` is rounding on held registers. In float64 the part on coordinates that change falls to 1e-34 by d = 8 and 1e-90 at d = K, a roughly exponential decay. That is the Bengio, Simard and Frasconi picture.
- **Weight gradient by pass.** Weights are shared across passes, so I split the weight gradient by the pass it comes from. After the readout and the last pass, the squared share left is 1e-14 to 4e-13 on `anylen` and 6e-9 to 1e-8 on `ntt`. Effective depth is 1 pass at every length. On the working Neural GPU it is also 1 pass (residual 3e-5 to 9e-5). On random controls and the failed seed it is the whole run.
- The top singular value of d(answer logits)/d(state) falls from 180 at d = 1 to about 1e-6 by d = 4 on `anylen`, and is 1e-5 to 1e-7 on `ntt`. The trained Neural GPU stays at 0.2 to 0.7. Its state Jacobian does not vanish, even though its sensitivity along the information direction does (next table).

### The 1994 test

For each problem, a second problem differs in one input digit. Its state at the same pass gives the direction Delta of real information. P is the share of the true answer change that a first-order prediction at the state sees. alpha* is the smallest share of Delta that changes the answer.

| net | max median P at d >= 8 | alpha = 1 gives the second answer | median alpha* at d = 4 to 64 | frozen read |
|---|---:|---|---|---|
| `anylen`, all n | 9e-29 to 7e-28 | 16/16 every cell | 0.1, flat with distance | `mixed` |
| `ntt` n = 2, 8, 32 | 2.5e-8, 2.3e-15, 8.8e-10 | every cell | 0.1-0.15 near the ends, 0.001-0.05 mid-run | `mixed` |
| Neural GPU s1, s2 | 4e-4 to 1.5e-3 | every cell | 0.4-0.5 | `mixed` |
| Neural GPU seed 3 (failed) | 0.7 to 1.3 | every cell | 0.1 to 1.4 | `absent` at n = 8, 20; `mixed` at 64 |
| random `ntt`, random Neural GPU | 0.6 to 1.3 | every cell | 0.45-0.65, or never | `absent` |
| random `anylen` | undefined: the input doesn't reach the answer numerically | | | `undefined` |

The frozen rule needs four things. Two hold on every written and working-trained net: P at most 0.01 and a flip at some alpha < 1. The full counterfactual state also gave the second answer on every problem (`verify.py` checks all 41 cells). The fourth, small changes `erased` from the state to 1% at alpha 0.02, held on no net. Post hoc, I looked at why. On `anylen` the working registers do erase it, to 0.3-0.6% of the initial size, while the copies of the input digit that the network holds unchanged keep it exactly (r = 1.000). On `ntt` registers that later passes don't read keep it. On the Neural GPU 8-22% stays. None of it reaches the answer: below alpha*, the share of the real answer change that reaches the logits is 0 to 2e-6 on written and trained nets, against 0.05-2.7% on random controls and the failed seed.

So I read it as a narrower version of the 1994 statement. A state that latches its answer hides small upstream changes from the answer and from the gradient, while large ones flip it. On these written nets the state does not literally erase everything. The trained Neural GPU shows the same pattern, at a weaker level (P about 1e-3 at d >= 8, falling 2 to 5 times per pass over the first passes).

Two further facts from the perturbations:
- **Protection covers the computing registers only.** On `anylen`, snapped registers (S, Zu, Yu, P, Ans) take 0.2 to 0.6 units before flipping and erase most smaller changes. The unsnapped reference coordinate flips from 0.05 and is never erased; the wave flags Lsep and Fz flip from 0.05 to 0.1 when every token is perturbed. On `ntt` the one-hot stage flag flips at 0.001. Isotropic noise of 0.001 times the size of a real information change flips `ntt` answers 90 to 96% of the time (pooled over distances d >= 1).
- **The gradient points at the unprotected coordinates.** On `anylen`, a step along the gradient of 0.01 times the information size flips 18-40% of answers, where isotropic noise of that size flips 5-8%.

This does not test training from random, and it is not the 1994 theorem, which concerns hyperbolic attractors and asymptotics. One `ntt` register cell (n = 32) was cut for cost; registers at n = 2 and 8 are complete.

## Experiment 2: same forward computation, different backward rule

Damage cells: noise on all program weights (`n1`, `n2`), one or four program weights zeroed (`z1`, `z4`), one scaled by 1.1 (`s1`), and noise on the output LayerNorm only (`p`, the positive control). Factors: loss (`ce`, `hinge` with margin 0.5, which is exactly zero on a clean net), backward rule (`std`, or `sur`: the GELU derivative replaced by a sigmoid of width tau times the unit's input scale, forward unchanged, so a straight-through rule through snapping), and support (`all`, or `nz`, the program's support). The grid was cut for cost before any dev result (amendment 2): 500 steps, batch 32, one dev draw. 12 scored draws per cell, every arm on the same draws.

**At the damaged point, before any step** (12 draws per cell). The cosine between the negative gradient and the true repair direction is 2e-10 to 2e-9 on `anylen` with `std`, except `s1` at 0.004. With the surrogate it is 5e-12 to 4e-6. On `two` it is 5e-14 to 1e-7, except `s1` at 2e-4 overall and 0.03 on the support. The surrogate moves a lesioned `anylen` weight from a median rank of about 17,500 to 8,400-9,800 of 139,440 by gradient size.

With denominators (amendment 1): on the clean `anylen`, 52% of the squared gradient is on zero entries, but per coordinate a program weight gets 12 times the rms of a zero entry (82 against 6.8). So there the gradient does not point away from the program. At the damaged points it depends on the damage. After noise or output damage, program weights get 2 to 13 times the zeros' rms. After a lesion, the zero entries get 18 to 300 times the program weights' rms (`z1`, `z4`); after a scaled weight, about the same (`s1`). On `two` the zero entries get 2 to 230 times the program weights' rms in every cell.

Along the straight line from damaged to clean weights, accuracy on lesions stays flat until beta 0.9 and comes back only at 1.0. After noise it rises gradually.

**Repair, frozen arms.** 0 of 1,152 scored runs restored (2 nets x 6 cells x 8 arms x 12 draws), and no comparison passes in any factor (0 of 48 cells each for backward rule, loss and support). On dev every lr broke these nets (on `two` dev, accuracy was at 0 by step 100 even at lr 1e-7), so the selection rule fell to its tie-break and picked the smallest lr, 1e-7, in every frozen arm. In scored runs the kept checkpoint is the starting point in 993 of 1,152, and no run gains more than 0.055 in strict accuracy at length 4. That includes the positive control `p` in every frozen arm, so the frozen control fails on both nets (best restored share 0.000 against the required 0.5).

**Repair, only the damaged entries free (post hoc oracle, amendment 3).** Restored of 12, scored draws, 500 steps. In brackets, the 1,500-step batch-64 check on `anylen` (192 runs), with the same lr and tau.

| net | cell | `ce` std | `ce` sur | `hinge` std | `hinge` sur |
|---|---|---:|---:|---:|---:|
| `two` | `p` | 12 | 12 | 1 | 1 |
| `two` | `z1` | 2 | 2 | 2 | 1 |
| `two` | `z4`, `s1` | 0 | 0 | 0 | 0 |
| `anylen` | `p` | 3 (7) | 3 (7) | 4 (8) | 4 (8) |
| `anylen` | `z1` | 0 (0) | 5 (6) | 0 (0) | 5 (6) |
| `anylen` | `s1` | 3 (5) | 6 (7) | 3 (4) | 5 (6) |
| `anylen` | `z4` | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

The one cell where the backward rule matters is `anylen` `z1`, a single program weight zeroed. With only that entry free, the standard gradient restores 0 of 12 at every lr from 1e-7 to 1e-2 under both losses (post hoc sweep, amendment 5). The surrogate restores 3 to 5 of 12 at lr 1e-3 and 1e-2. At 1,500 steps it is 6 of 12 against 0, with an uncorrected McNemar p of 0.03. `s1` leans the same way (5-7 against 3-5) and does not pass a test. On `two` the surrogate makes no difference.

**Read.** The frozen control failed, so by the protocol's rule the frozen null cannot be read as a property of the signal. What the oracle arm adds: once nothing else moves, plain gradient restores damage at the output layer and some single-weight damage, and the surrogate changes nothing that passes a test. That points at the update spilling onto other tight weights as the main obstacle, more than at the plateaus' zero derivatives. Nothing here is a repair a real learner could do, since it doesn't know which entries were damaged.

## Experiment 3: one-step against free-running on CLRS quicksort

All 30 TEACH-WRITTEN `best` checkpoints and the written teacher, on the official-style test sets (seed 3: 32, 32, 8 instances at n = 16, 64, 128) and a fresh set of the same sizes (seed 20261107). The replay gate passed: free-running F1 equals TEACH-WRITTEN's on 87 of 90 comparisons, the other 3 within 0.0034, the same 3 that TEACH-WRITTEN's own replay reported. Means over checkpoints, seed 3; the fresh set agrees within 0.03 on every column in the table below.

| arm | n | F1 free | F1 reference-fed | F1 hidden zeroed | reading | one step | free-running, all steps | reference-fed, all steps | gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `hints` (10) | 16 | 0.992 | 0.922 | 0.656 | 0.768 | 0.591 | 0.612 | 0.757 | reading, one step, keeping |
| `hints` | 64 | 0.787 | 0.502 | 0.183 | 0.307 | 0.261 | 0.113 | 0.353 | all three |
| `hints` | 128 | 0.414 | 0.276 | 0.054 | 0.158 | 0.153 | 0.057 | 0.198 | all three |
| `lathints` (5) | 64 | 0.458 | 0.393 | 0.087 | 0.431 | 0.272 | 0.121 | 0.451 | all three |
| `lat` (5, probes) | 64 | 0.506 | 0.160 | 0.080 | 0.369 | 0.102 | 0.132 | 0.317 | all three |
| teacher | 16, 64, 128 | 1.000 | 0.91-0.99 | 0.91-0.99 | 1.000 | 0.12-0.14 | 1.000 | 0.66-0.71 | one step |

`reading` is accuracy on p, r, i, j where the target does not change, `one step` where it does, both reference-fed with the model's own hidden state. All arms and sizes are in TABLES.md.

- **Where it fails.** Already at the first two gates, at every size. From the true hints, the official-hints student takes 59% of changing steps right at n = 16, 26% at 64 and 15% at 128. Reading unchanged values also falls with size (0.77, 0.31, 0.16). So the size failure is not only accumulated drift: one step from a correct state is already wrong.
- **Hard reference hints hurt output.** Reference-fed output F1 is below free-running everywhere (0.992 to 0.922 at n = 16, 0.787 to 0.502 at 64). The checkpoints trained with their own soft hints fed back and never saw hard ones, which is the Professor Forcing caveat: these scores are biased by distribution shift, and in this case downward.
- **Keeping.** At n = 16 and 64, errors do not compound worse than independent one-step errors: actual free-running survival is at or above the product of reference-fed one-step rates. At n = 128 it falls below that product by step 4 (0.025 against 0.054), and both are near zero by step 8. At n = 64, after a reference-fed first half, accuracy drops from 0.21 to 0.13 at the first free step and to 0.03 within 128 steps. Feeding true hints again after a free first half brings it back to 0.16-0.20 (reference-fed on the same steps: 0.21 to 0.27). So the free-running state is mostly not damaged beyond what the hints fix.
- **The recursion stack.** Pop steps, where the next interval comes from pending recursion, are 2.6-4.0% of steps and 24-30% of interval changes. A local rule using only current hints and keys gets 51-72% of them, which is a lower bound for a memoryless reader. By the frozen rule, the `hints` arm keeps pending ranges in its hidden state at n = 16: p, r at pop steps 0.215 with hidden against 0.008 zeroed, difference 0.21 [0.11, 0.31]. That reading is weaker than the rule makes it look: hidden helps more on ordinary scan steps (+0.36) and child steps (+0.64) than on pops. No arm clears it at n = 64 or 128. What reference-fed evaluation cannot show: the student executing quicksort, since every step gets a state it did not make, and the stack is never supplied.
- **The teacher's state is not the hints.** Free-running, the written teacher is exact. Fed true hard hints with its own hidden state, it gets 12-14% of changing steps right; given its own free-running hidden state too, 81-86%. The rest of its state lives in about e^-30 of soft mass on the successor in its fed-back `pred_h`, which hard hints drop. Giving the matched students the teacher's hidden state (mode `T`) does not help them either: `lathints` all-step accuracy 0.71 with its own hidden, 0.49 with the teacher's. Matching to nRMSE 0.6-0.7 in TEACH-WRITTEN did not put them in the teacher's coordinates.
- **Against ALTA** (Shaw et al. 2025): its trace supervision gives the full program state per sub-layer, and its local-optimum theorem is for that per-step loss. CLRS hints are a strict subset of the state, as both the stack and the teacher's soft mass show. Under official hints, neither a student nor the written network takes reliable steps from the hint state alone.

## Experiment 4: does a scaled network survive training?

Added at the user's request, with its protocol and read frozen before running (amendment 4). `scaled` multiplies `lnf.w` and `lnf.b` by 5.2 (`anylen`) or 6.72 (`two`), which is exactly that factor on the logits. It keeps 100% strict at CE 0.0097 and 0.0101. `lnf.w` alone times 1.1 breaks both nets (0.09 and 0.01 strict), because the readout bias holds a quadratic -j^2 decode, so scaling only the gain moves its peak. That explains the earlier lanes' "no simple temperature direction": the direction exists and needs both tensors. Gradient norm at the scaled point: `anylen` 3,504 to 323, `two` 2.9e8 to 1.75e8, so on `two` the gradient barely depends on confidence.

| net | eps* along the CE gradient, base / scaled | ratio [95% CI] | eps*, random direction | eps*, gain direction | cosine of CE gradient with gain direction | CE gained along the gradient before breaking, scaled |
|---|---|---|---:|---:|---:|---|
| `anylen` | 2.61e-7 / 2.73e-7 | 0.996 [0.935, 1.08] | 7.9e-7 | > 1e6 | 3e-10 / 8e-11 | 0.0098 to 0.00973 |
| `two` | 4.87e-11 / 4.87e-11 | 1.00 [1.00, 1.00] | 2.6e-8 | > 1e6 | 7e-14 / 2e-15 | 0.0095 to 0.0085 |

eps* is the largest L2 weight-space distance with every problem still right, median over 8 gradient batches. `anylen` fails first at length 100 along the gradient.

- **SGD, all weights.** Both conditions die at every lr tried. Scaled `anylen` survives more steps at some small lr (100 against 25, 700 against 100), as its 11x smaller gradient predicts. But its displacement before failure is smaller, not larger: R_D = 0.39 [0.10, 0.68] over 12 pairs. `two` with all weights dies by step 25 at every lr in both conditions, so R_D is undefined there.
- **Frozen embeddings on `two`.** SGD survives 2,000 steps at lr 1e-17 to 1e-15 in both conditions. At 1e-14 the base net fails and the scaled one survives. CE falls to 0.003-0.004 scaled and 0.54-0.64 base, the earlier lane's frozen-embedding result. Adam, with the caveat that it normalizes gradient scale: dead by step 25-100 at every lr, except `two` with frozen embeddings at 1e-10 and 1e-9.
- **Frozen read: `mixed` on both nets.** On `anylen`, the R_D interval falls below 1/3, in the direction opposite to `confidence implicated`. On `two`, R_D has no pairs. Neither net comes near `confidence implicated` (that needs a lower bound of 3). Read plainly, this is sensitivity: removing the confidence pressure lets neither net take a larger gradient step. The gradient has essentially no component along the safe direction that buys confidence, and a step along it breaks the net at the same distance as before, at a distance smaller than a random direction allows (3x smaller on `anylen`, 500x on `two`).

This starts from correct networks. It says nothing about reaching one from random.

## What this does not show

- Nothing here is about training from random, except the trained Neural GPU's state, which is measured, not trained here.
- Experiment 1 tests two written nets and one trained architecture at fixed states. The tension read is `mixed` by the frozen rule. My narrower reading of it is post hoc.
- Experiment 2's surrogate is one family (sigmoid in place of the GELU derivative) at one optimizer and 500 steps; the 1,500-step check covers only the oracle arms on `anylen`. The positive control failed in every frozen arm, so the frozen null cannot be read as a property of the gradient signal.
- Experiment 3 uses frozen checkpoints trained on soft self-fed hints and evaluated on hard reference hints. Both modes are off their training distribution in different ways.
- Experiment 4's frozen read is `mixed`. "Sensitivity" is my reading of the measurements, not the rule's verdict.

## Prior art

[PRIOR_ART.md](PRIOR_ART.md). The tension is Bengio, Simard and Frasconi 1994, and this lane measures it on networks with known registers. Nothing here is a new effect. I found no earlier report of the first-order share P on such networks, and no earlier attempt to repair hand-written networks by gradient; one search each. Teacher forcing against free-running is Williams and Zipser 1989, scheduled sampling, Professor Forcing and DAgger. The stack point is Jürß, Jayalath and Veličković's Recursive Algorithmic Reasoning. ALTA (Shaw et al. 2025) is compared in Experiment 3; I read it through a summary, not the full text. The spurious line in the MULT-LOGDEPTH report about the Neural GPU's saturation pull is not reused.

## Changes from the protocol

All dated in [PROTOCOL.md](PROTOCOL.md), each saying what had been read when it was written.
1. Coordinator additions (denominators, output-gain control, ALTA), before any Experiment 2 result.
2. Experiment 2 grid cut for cost (500 steps, batch 32, lr 1e-7 to 1e-3, tau 0.5 and 2, 1 dev draw). Only the run rate had been read.
3. Positive control failed on dev for `two`; the oracle `dmg` support was added post hoc.
4. Experiment 4 added at the user's request, frozen before running.
5. Post hoc lr sweep on the one cell where the surrogate differed (`anylen` `z1`, oracle support), after all scored results.
- Ceiling raised from $25 to $100 by the user via the coordinator (logged in DECISIONS.md, not a protocol change). It paid for the 1,500-step check of the oracle arms and the lr sweep. No new scope.
- Not run, or cut: the `ntt` n = 32 register cell; `ntt` n = 64. Two Spot preemptions and one GPU out-of-memory in Experiment 3, and one preemption in Experiment 2, were resumed from saved output. No scored number depends on a partial run.

## Cost and closure

Planning cost, rate x observed VM life + $0.50 per VM, from `resources.json`, not an invoice. **$16.76 of the $100 ceiling** (raised from $25; the lane would have finished inside $25).

| Item | VMs | VM hours | $ |
|---|---:|---:|---:|
| Experiment 1, n2-highcpu-32 Spot | 1 | 2.3 | 1.88 |
| Experiment 2, n2-highcpu-32 Spot (dev, scored, diagnostics, oracle arms, long check, lr sweep; 1.6 h discarded after the grid cut) | 7 | 10.5 | 9.83 |
| Experiment 3, L4 Spot (g2-standard-8) | 3 | 4.5 | 3.77 |
| Experiment 4, n2-highcpu-32 Spot | 1 | 1.3 | 1.29 |
| Total | 12 | 18.7 | 16.76 |

Two L4 requests were refused for stockout and cost nothing. Spot preemptions: 2 in Experiment 3, 1 in Experiment 2, each resumed from saved output. All 12 VMs are deleted. Raw output stayed in cloud storage and the small JSON in `results/`. Nothing pushed, posted or published. Local Python runs were the smoke tests, `gain.py`, `gain2.py`, `posthoc_e1.py` and `verify.py`, each under 60 s on one thread.

## Where each number lives

- Experiment 1: `results/e1a/e1/*.json`, tables by `analyze_e1.py`; post hoc check `posthoc_e1.py` to `results/posthoc_e1.json`.
- Output gain: `gain.py`, `gain2.py` to `results/gain*.json`.
- Experiment 2: draw selection `results/e2sel/`, diagnostics `results/e2diag/`, dev `results/e2vt/`, `e2va/`, `e2dt/`, `e2da/`, scored `results/e2scored/`; `analyze_e2.py`.
- Experiment 3: `results/e3/*.json`, `analyze_e3.py`.
- Experiment 4: `results/e4s/scan/`, `results/e4t/train/`, `analyze_e4.py`.
- Code: `code/e1.py`, `e2.py`, `e3.py`, `e4.py`; job specs `specs/`; cloud control `cloud.py`; spend `resources.json`.
