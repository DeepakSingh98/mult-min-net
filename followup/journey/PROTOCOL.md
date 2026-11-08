# WSP-JOURNEY protocol (frozen before any run)

Written 2026-11-08, before any code for these experiments ran. Budget: $100 on GCP, set by the user. Measurement
only: no weights are written by an editor. Small local smoke tests (60 s, one thread) check that code runs and are
logged in DECISIONS.md; they are not scored.

## Why

The follow-up post says: start training from a network that already works, and gradient descent walks away from it.
Two questions the post can't yet answer:

1. **Does gradient descent keep a destination it found itself?** If it keeps trained solutions but leaves written ones,
   the problem is solutions the gradient can't see, not solutions in general.
2. **Is the any-length network's fragility caused by the loop?** That network breaks first on the longest problems
   (3e-6 along the gradient at 4 digits, 2.6e-7 at 100). One reading: the gradient sees only the last pass, but the
   weights are shared, so the same change is applied at every pass and compounds. Untying the passes tests that.

## Networks

| id | what | found by | strict set (length, problems), checked in this order |
|---|---|---|---|
| `anylen` | 870-weight looped transformer, decimal multiplication, any length | written (Opus) | (4, 256), (20, 32), (100, 4) |
| `two` | 487-weight two-layer transformer, 4-digit decimal multiplication | written (Astra) | (4, 256) |
| `ng1`, `ng2` | Neural GPU `ng-bin20-64-0.0005-rx1` seeds 1, 2, binary multiplication, trained on lengths up to 20 | gradient descent | (20, 256), (8, 128), (64, 32) |
| `tf` | looped transformer `tf-add-soft-s1` (the any-length architecture family: causal, ALiBi, no position table), decimal addition, trained up to 8 | gradient descent | (8, 256), (4, 128) |

`strict` = every answer digit of every problem is the argmax (teacher-forced for `tf`, as its lane scored it). Problems
come from fixed seeds, so every check uses the same problems. A network must be strict on its whole set at the start.
If a length fails at the start, it is dropped from that network's set and the drop is reported. The two written nets
use exactly the WSP-STATE-SIGNAL Experiment 4 problems and code paths.

## Conditions (confidence)

- Written nets: `low` = as written (CE about 0.52 and 0.66). `high` = final layer norm weight and bias scaled together
  (anylen x5.2, two x6.72; CE about 0.01), as in Experiment 4.
- Trained nets: `high` = as trained. `low` = readout scaled down by one factor alpha < 1 so that plain CE on the
  gradient batch (seed 7700) is 0.52 (bisection). For `ng` that's the output conv weight and bias; for `tf` it's the
  final layer norm weight and bias. Scaling logits by alpha > 0 never changes an answer.

The `low` condition matters most. It takes a trained net off its resting point, so it is pushed by the loss as hard as
the written net is. If it still stays put, the difference is in how the solution is built, not in where it sits.

## Experiment 5: does gradient descent keep its own destination?

**Scan (no training).** At each net and condition, 8 draws (seeds 7700-7707). For each draw, a unit direction in the
space of all parameters:
- `grad`: minus the gradient of plain CE on 64 problems at the gradient length (anylen 4x4, two 4x4, ng 20x20 binary,
  tf 8x8);
- `own` (ng only): the same with the training loss (CE with label smoothing 0.1), dropout off;
- `rand`: standard normal over every entry;
- `gain`: the readout-scaling direction above.

eps* = the largest L2 step with every strict problem still right: grid 10^(k/2) for k = -16..12, then 8 bisection
steps (Experiment 4's procedure). The first failing length is recorded. Also: the cosine between `grad` and `gain`.

**Training.** From each net and condition, Adam at lr 1e-6, 1e-5, 1e-4, 1e-3, plain CE, batch 64, the net's own
training length distribution, 3 seeds, 2,000 steps. Adam is the primary optimizer here because it removes gradient
scale, which differs by orders of magnitude across these nets. Trained nets also run their own recipe at both
conditions, 3 seeds:
- ng: Adam lr 5e-4, eps 1e-3, clip 4, label smoothing 0.1, dropout on, batch 32;
- tf: Adam lr 1e-3, clip 1, batch 64.

Strict at the first length every 25 steps, the full set every 100 steps, stop at the first failure. Recorded: the
failure step and the relative displacement ||W - W0|| / ||W0|| at the last passing check.

**Predictions.**
- **P1.** For every trained net at both conditions, R = median eps*(grad) / median eps*(rand) >= 1. For the written
  nets, R <= 0.33 (Experiment 4: anylen 0.33, two 0.0019).
- **P2.** At `low`, cos(grad, gain) >= 0.1 for every trained net, and <= 1e-6 for the written nets.
- **P3.** Every trained net at both conditions survives 2,000 steps at all 3 seeds with Adam at lr <= 1e-4, and with
  its own recipe. Every written net at both conditions fails at every lr, at all 3 seeds.

**Reading, fixed now.**
- **"Gradient descent keeps its own destinations and walks away from the written ones":** P1 and P3 both hold.
- **"No difference":** a trained net at `low` has R <= 0.33, or fails P3's Adam cells at lr <= 1e-4 in at least 2 of
  3 seeds.
- **"Mixed":** anything else, and say which part held.

## Experiment 6: untie the loop

`anylen` only. **Untied** = the last J = 22 passes each get their own copy of both blocks, counted from the end:
copy j runs pass K-1-j, and every copy starts at the written weights. Earlier passes, which only occur on problems
longer than 4x4, use the written weights. `wte` and the final layer norm stay shared. At the start, the function is
identical to `anylen`. **Tied** = `anylen` as written, rerun in the same code.

**Scan.** Base condition, the same 8 gradient seeds and the same strict set as Experiment 5. Directions:
- `grad` (plain CE, 4x4, 64 problems; untied: the gradient for every copy and shared tensor);
- `rand` over the same support;
- `gain`.

Also recorded, not scored: each copy's share of the untied squared gradient.

**Training.** Plain CE, the Experiment 4 grid (SGD lr 1e-12 to 1e-4 by decades; Adam 1e-10 to 1e-3), 3 seeds, 2,000
steps, all parameters, pairs n, m in 1..4, batch 64, tied and untied. The reading uses SGD. Adam rescales every
coordinate, so it moves the early copies even where their gradient is tiny, which would hide the effect being tested.
Adam is reported but not read.

**Predictions.**
- **P4.** Median eps*(grad) untied >= 5 x tied, and under the untied step the first failure is not at length 100 in at
  least 6 of 8 draws.
- **P5.** Untied SGD survives 2,000 steps at all 3 seeds in at least one lr cell where tied fails.

**Reading, fixed now.**
- **P4 holds:** the length-dependent fragility along the gradient comes from the same change being applied at every
  pass. The gradient sees one pass, but the shared weights carry its change into all of them.
- **Untied within 2x of tied:** the fragility lies in the last passes themselves, and the loop is not the cause.
- **Between 2x and 5x:** mixed.

## What this does not test

Training from random. Other trained networks. Whether any of this holds in large models. Experiment 6's untied copies
are one way to untie, aligned from the end.

## Cost and closure

GCP n2-highcpu Spot VMs through `cloud.py`, lease-bounded, deleted at the end. Planning cost is recorded in
`resources.json`. Raw output goes to cloud storage, and only small JSON comes back
to this Mac. Nothing is pushed, posted or sent.
