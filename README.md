# Four-digit multiplication in a few hundred nonzero weights

Three small GPT-2-style transformers that multiply two 4-digit numbers, with every weight written by a program rather
than learned. GPT-6 Astra wrote the programs. The smallest network we found in each architecture arm:

| Network | Config | Nonzero weights | Dense size | Strict wrong, 1M fresh-seed problems (fp32 / bf16-rounded) | Strict wrong, all 81M pairs (fp32 / bf16-rounded) | Min strict margin |
|---|---|---:|---:|---|---|---:|
| Two layers | 8 heads x 2, d_model 512, d_mlp 69 | **487** | 231,658 | 0 / 0 | pending | 0.358 |
| One layer | 24 heads x 4, d_model 96, d_mlp 161 | **927** | 72,257 | 0 / 0 | pending | 0.0156 |
| One block, looped 42 times | 2 heads x 7, d_model 50, d_mlp 44 | **364** | 9,386 | 0 / 0 | pending | 0.936 |

- **Strict** means: at every one of the 8 answer positions, the correct token's logit is strictly above all 13 other
  tokens' logits. A tie counts as wrong.
- **bf16-rounded** means every weight rounded to bfloat16, with the computation still in float32. Native bf16
  inference was not tested.
- The fresh seeds were drawn from `os.urandom` after the networks were delivered: 559835402 (two layers), 2376587163
  (one layer), 1761220234 (looped). Rerun any of them with `python verify.py weights/NAME.npz --n 1000000 --seed S`.
- The min margin is the smallest (correct logit - best other logit) over the problems checked, in either precision.
- These counts are upper bounds: the smallest networks **we found**, not proven minima.

## The task

The task is a x b for a, b in 1000..9999, written as one 20-token sequence, least-significant digit first, one token
per digit, with no separators:

    position:  0  1  2  3  4  5  6  7  8  9   10  11  12 13 ... 19
    token:     a0 a1 a2 a3 *  b0 b1 b2 b3 EOS H1  H2  c0 c1 ... c7

The product c is zero-padded to 8 digits. The logits at position 11 + k predict answer digit c_k.

Scoring is teacher-forced: the earlier answer digits c_0..c_(k-1) are visible as input. For a network that gets every
digit right, this is the same as greedy decoding. The format and the network family follow Bai et al., *Why Can't
Transformers Learn Multiplication? Reverse-Engineering Reveals Long-Range Dependency Pitfalls* (arXiv 2510.00184),
which trains a 2-layer, 4-head, 768-wide GPT-2 on it.

## The network

[arch.py](arch.py) is the whole model, in about 60 lines of PyTorch. It is a GPT-2-style decoder:

- pre-LayerNorm blocks with causal multi-head attention and a gelu_new MLP;
- learned position embeddings and a final LayerNorm;
- an output layer tied to the input embedding, and a 14-token vocabulary (10 digits, `*`, `EOS`, `H1`, `H2`).

Each network chooses its own sizes: `n_layer`, `loop`, `n_head`, `d_head`, `d_model` and `d_mlp`. Unlike stock GPT-2,
the attention width n_head x d_head is chosen independently of d_model. With `loop` = K, the stack of blocks is
applied K times with the same weights.

The three arms constrain only the shape:

| Arm | Rule |
|---|---|
| Two layers | n_layer = 2, no looping |
| One layer | n_layer = 1, no looping |
| Looped | n_layer = 1 applied K <= 64 times; the shared weights are counted once |

## The rules

Every weight is set by code: the programs in [build/](build/). No gradients, training, optimizers or iterative
fitting were used, and there was no automated search over weights or pruning masks (no loop that tries several
settings and keeps the best one). Closed-form computation was allowed if disclosed. An independent read of every
executed code block and every build program found no violations.

**Score.** The score is the number of nonzero weights over all tensors. LayerNorm gains and biases count, and the tied
embedding counts once. The dense size of each configuration is much larger. A sparse count does not mean the same
saving in runtime, file size or FLOPs.

## Reproduce

```bash
pip install -r requirements.txt
python make_weights.py                  # rebuild all three networks from build/ and check sha256 against weights/
python verify.py weights/*.npz          # 100,000 random problems each, strict, fp32 and bf16-rounded
```

- `make_weights.py` runs each `build/NAME.py` in a fresh process and packs its output with `arch.save()`. The result
  must match [weights/SHA256SUMS](weights/SHA256SUMS) byte for byte. This was checked in a fresh clone.
- `verify.py` takes `--seed` for a reproducible run; without it, it draws a seed from `os.urandom` and prints it.
  - `--exhaustive` checks all 81M operand pairs. That takes hours on one machine; `--a-from` and `--a-to` run one
    slice.
  - Single-threaded on a laptop, 100,000 problems take about 20 s (one layer), 30 s (two layers) and 2.5 min (looped,
    42 passes) per precision. Use `--n 20000` for a quick look.
- `weights/NAME.npz` is a plain zip of `.npy` arrays: one dense float32 array per tensor, plus `config` as a JSON
  string. `arch.load()` reads it, and `np.load` works without pickle.

## How each network works

All three rely on the same identity. With the lower answer digits visible, the quantity

    q_k = sum_{i+j<=k} a_i b_j 10^(i+j-k) - sum_{j<k} c_j 10^(j-k)

is an integer whose last digit is c_k. So each network reads the answer digits it has already written, rather than
computing the carry chain from the operands alone.

Each network's full write-up is in [explanations/](explanations/), written by Astra and lightly edited.

**Two layers, 487** ([explanation](explanations/two_layer.md))
- *Block 0 attention:* 8 heads of width 2 fetch a_i and b_(k-i) at each answer position, using a quadratic selection
  score.
- *Block 0 MLP:* products come from the curvature of gelu_new near zero,
  C [g(e(a+b)) - g(e(a-b))] ~ 4 c e^2 C a b. That takes two neurons per product plus one shared correction, and
  gives the column sums.
- *Block 1 attention:* three heads read the previous two columns and answer digits. This gives the column value up to
  an omitted carry of at most 0.35.
- *Block 1 MLP:* 34 pairs of GELU "reset" neurons subtract 10 per decade. One extra hinge handles the single case
  9999 x 9999.
- *Readout:* the tied readout gives digit d the logit 2 d r - d^2 + 1, so the nearest digit wins. The +1 keeps every
  digit above the marker tokens.

**One layer, 927** ([explanation](explanations/one_layer.md)). Attention does the multiplying.
- *Attention:*
  - 7 heads average uniformly over digit pairs.
  - 16 heads take slightly digit-biased averages. The softmax curvature gives
    z ~ (x+y)/2 + (x-y)^2/2048, and with x^2 and y^2 as linear embedding features this yields x y.
  - One head reads the visible answer prefix with scores j log 10.
- *Output projection:* forms each column's reduced integer.
- *MLP:* 161 GELUs take that integer mod 10 as a sawtooth.
- *Readout:* one digit-only bias in the final LayerNorm puts digit 0 exactly 2^-6 above the markers. This is the
  network's smallest margin (0.0156).

**One block looped 42 times, 364** ([explanation](explanations/looped.md)). The shared block runs as a small program.
A counter and two "squared clock" coordinates select what each pass does.

| Passes | What they do |
|---|---|
| 1–4 | a three-GELU multiplier, -C [g(s(a-b)) - g(s a) - g(-s b)] ~ k a b, builds the 16 digit products |
| 5 | routes signed products and the visible answer digits into columns |
| 6 | gathers a coarse, carry-corrected column integer |
| 7–41 | two GELUs subtract 10 whenever the value is above 9.375; 35 passes cover values up to 350 |
| 42 | tied quadratic readout with strict separation from the markers |

Every weight of the looped network is exactly representable in bfloat16, so its bf16-rounded test uses identical
weights.

**bf16.** All three were designed for bf16-rounded weights. Most weights are powers of two or short dyadic fractions.
Coefficients that must cancel are built from bf16-exact factors, so rounding preserves the cancellation.

[figures/](figures/) has three views of the two-layer network: every weight matrix, the attention patterns of the
active heads, and its MLPs and readout across 4,000 problems. The script that drew them is not included.

## Caveats

- **Upper bounds, not minimums.** No lower bound was established. Astra's own guesses about how much smaller each could
  go are in the explanations.
- **Carries come from the written answer.** Every network reads the earlier answer digits, which the task supplies
  (teacher forcing, equivalent to greedy decoding). None computes a x b from the operands alone.
- **bf16 means rounded weights only.** The bf16 numbers use bf16-rounded weights with float32 computation. Native bf16
  inference was not tested.
- **Sampled vs exhaustive.** The 1M-problem results are random samples. The exhaustive column covers every operand
  pair once, on one software stack (PyTorch 2.5.1, CPU). Neither is a proof for other implementations.
- **Small margin.** The one-layer network's margin is 2^-6. A different softmax or GELU implementation, or GPU kernels,
  could move logits by more than that. The other two have margins 0.36 and 0.94.
- **Redesign, not from scratch.** Astra designed each network while able to see its own earlier, larger build for the
  same task. The earlier builds were two-layer 768-wide networks with 224,420 nonzeros, then argmax-only versions with
  511 / 1,088 / 438. Each `build/` program reads only `arch.py`, so the shipped weights do not depend on the old ones,
  but the design process had the earlier work to learn from.
- **One editor.** All three networks are Astra's (GPT-6 Astra via `codex exec`, effort high, CPU sandbox, at most 120
  Python runs per session). Parallel Claude Opus sessions were cancelled early and produced no network, so this is
  not a comparison between editors.

For context: trained by ordinary SGD/AdamW, the 768-wide network of Bai et al. reached 1.5% exact (14% with dropout 0)
after 300k further steps from its plateau in our earlier runs. Hand-written 768-wide builds reach 100%. Those numbers come from an earlier experiment
and are not part of this repository.

## Files

```
arch.py            task encoding, model forward, weight-file format
build/             one program per network (Astra's, unchanged except the path to arch.py)
weights/           the three networks (.npz) and SHA256SUMS
make_weights.py    rebuild from build/ and check sha256
verify.py          strict check on random problems or all pairs, fp32 and bf16-rounded weights
explanations/      how each network computes (Astra's write-ups)
figures/           plots of the two-layer network
```

## Credits

- The network programs and the explanations were written by GPT-6 Astra.
- The task format and network family come from Bai et al., arXiv 2510.00184.
