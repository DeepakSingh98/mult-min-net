# Hand-built strict-margin multiplication transformer

> Astra's own write-up of this network, lightly edited for this repository: its in-session test tables (seeds it
> chose itself) are replaced by a pointer to the README, whose results were measured separately after delivery;
> sandbox paths are replaced by repository paths; "parameters" is changed to "nonzero weights" where it counts
> nonzeros. "The supplied previous/earlier build" means Astra's own earlier, larger build for this arm (not shipped
> here). Everything else, including its guesses about the minimum, is Astra's text.

## Delivered network

- Configuration: `n_layer=1, loop=42, n_head=2, d_head=7, d_model=50, d_mlp=44`.
- **364 nonzero weights**; the dense size is **9,386**.
- Measured strict results are in the README.

Every delivered weight is already exactly representable in bfloat16.
Consequently the two acceptance modes use identical weights and float32
computation. This is NOT a claim about doing the computation itself in bf16.

The bounds below explain the design margin but are not a formal proof of
all float32 rounding errors for every input.

## Method and attribution

The arithmetic core is adapted from the supplied earlier-session 438-nonzero
builder. The new builder contains that logical construction and explicit
algebraic transformations; it does not import or read the previous builder,
saved weights, operands, or test data. Model definitions are
loaded only from `arch.py`.

There was no training, autograd, optimizer, fitting to examples, or automated
weight/mask search. No weights were calculated by regression or least squares.
The decisions use arithmetic identities, GELU's Taylor expansion, dyadic
coefficients, and manually diagnosed numerical failures.

The construction initially creates a paired logical circuit, quantizes its
constants, converts internal signals to single coordinates with mean-rejecting
projections, and packs the residual axes. Subsequent explicit transformations
produce the delivered circuit. Dead head and neuron slots are removed
structurally, not selected by empirical pruning.

## Residual coordinates and embeddings

Write `T(d)=d(d-1)/2`. Residual width stays 50 so that the anchor variance has
an exact power-of-two scale.

| Coordinates | Meaning |
|---|---|
| 0, 1, 2 | digit token code `d, -d, T(d)` |
| 3, 5, 7, 8 | position indicators for b digits, positions 5..8 |
| 4 | permanently zero mean-reference coordinate |
| 9..16 | indicators for prediction positions 11..18 |
| 17 | a-source position index j, zero elsewhere |
| 18 | a-source quadratic/validity key j^2-32, zero elsewhere |
| 20, 21 | anchor pair, magnitude A=5*2^22 |
| 22, 23 | stored squared-clock states |
| 24 | loop counter r |
| 25 | retrieved a-digit mailbox |
| 26..29 | four-product history |
| 30..36 | seven local signed product/answer columns |
| 37..43 | seven causal gathered columns |
| 44 | assembled column value and then modulo-ten remainder |
| 6, 19, 45..49 | unused zeros |

Digits have anchors `(+A,-A)`; markers have `(-A,+A)`.
All markers have zero digit/triangle codes.
Position embeddings contain twelve routing indicators and seven address-key
entries, for 19 nonzeros total.

With D=50, the anchor pair contributes standard deviation exactly `2^22`.
The active LN1 and LN2 gains are `2^22`, approximately preserving small signals.
The finite epsilon and signal-dependent variance introduce small deviations.

All nonconstant input projections reject the common LayerNorm mean:
their gain-weighted coefficients sum to zero. Where necessary, coordinate 4
supplies the cancelling reference coefficient. Its residual value remains zero,
so this changes no intended signal. The balanced routing and multiplier
difference neurons need no reference entry.

LN2 biases expose:
- `position_indicator - 1` at each of the twelve position coordinates;
- `(r-4)^2` and `(r-5)^2` at clock coordinates 22 and 23;
- `r-41` at the counter coordinate.

The final LN reads only the digit-coordinate difference with gains `2^24`.
Its triangle output is the constant -4, and its coordinate-20 output is
the constant `2^-24`.

## Attention head 0: serial digit retrieval

At a-source position j=0..3 the keys are `32*j` and `32*(j^2-32)`.
Other positions have zero keys.
The query components are `192-64*r` and `-32`.

The source score before division by sqrt(7) is

    1024 * ((6-2*r)*j - j^2 + 32).

The first four passes, r=0,1,2,3, therefore select a3,a2,a1,a0.
The quadratic maximum is separated by a very large score gap, and the +32
term excludes invalid source positions. Later passes select a0, harmlessly.
The source validity bonus and quadratic penalty share one key coordinate.

The head's single value component reads the digit and writes coordinate 25.
The MLP uses and clears that mailbox each pass. Early a-source positions have
shorter causal prefixes, but their products are never routed into answers.

## Attention head 1: causal uniform gather

All its query/key entries are zero, so it attends uniformly over the causal
prefix. Seven value components read local columns 30..36. Seven output weights
`222/256` write gathered columns 37..43.

Local columns remain zero until the fifth MLP pass. The sixth attention pass
therefore gathers exactly one copy of the routed information for assembly.
Later repeated gathers are irrelevant because the assembly clock is closed.

## MLP mapping and loop schedule

The same 44 units and same weights execute on all 42 passes.

| Units | Function |
|---|---|
| 0..2 | analytic three-GELU multiplier |
| 3 | clear retrieved a-digit mailbox |
| 4..7 | shift four-product history |
| 8..23 | route the 16 b-position/product-register combinations on pass 5 |
| 24..30 | route seven negative teacher-forced answer digits on pass 5 |
| 31..38 | assemble the eight prediction columns on pass 6 |
| 39..40 | recurrent subtraction of ten |
| 41..42 | final digit-code overwrite on pass 42 |
| 43 | shared linear update of both squared clocks |

### Passes 1..4: analytic multiplication

Let g be `gelu_new`, `s=2^-10`, `alpha=sqrt(2/pi)`, and `C=5*2^18`.
Near zero,

    g(x) = x/2 + alpha*x^2/2 + O(x^4).

The three units compute

    M(a,b) = -C * [g(s*(a-b)) - g(s*a) - g(-s*b)].

The linear terms cancel and the leading mixed term is

    kappa*a*b,  kappa=(5/4)*alpha=0.9973557010035817.

The first neuron's two input coefficients sum to zero, saving a reference.
C is the bf16 rounding of the analytic inverse quadratic coefficient.

The later route and gather factors supply

    R = (37/32)*(222/256) = 8214/8192,
    R*kappa = 1.0000341464896754.

These constants are analytic/dyadic calibration, not fitted to examples.
On all 100 digit pairs, the isolated float32 calibrated difference-form
multiplier had error from -0.0000576973 to +0.0019950867.
That isolated diagnostic does not bound every error in the complete network.

The history shift runs in parallel with the multiplier. After four passes,
each b_j position holds approximately

    [kappa*a0*b_j, kappa*a1*b_j, kappa*a2*b_j, kappa*a3*b_j].

### Squared clocks

Before pass r+1, the stored clock for target t is

    q_t(r) = r^2 - 2*t*r,  t=4 or 5.

Its LN2 bias t^2 exposes `(r-t)^2`.
Unit 43 reads raw r using the shifted counter plus a bias 41, and writes 2r
to both clocks. Output biases -7 and -9 provide `1-2*t`, so

    q_t(r+1) - q_t(r) = 2r + 1 - 2t.

The counter increments by one using a third output bias.
All updates are represented with one saturated-linear GELU unit.
The clocks begin at zero stored state, requiring no embedding entries.

### Pass 5: route signed columns

Each product-routing unit computes, before its GELU scale 32,

    product_i + 128*(position_indicator - 1) - 129*(r-4)^2.

The three variable coefficients sum to zero. At the intended position and
r=4 it passes the product; otherwise the positional or clock penalty exceeds
the largest digit product. Its output is scaled by 37/32 into column i+j.

Seven analogous units route each visible answer digit negatively into its
column. Causality ensures that position 11+k sees c_r only for r<k, never c_k.

The negative routing uses R rather than exactly one after the gather.
This creates only a small extra error in the weighted answer prefix.

### Pass 6: recover a coarse carry-corrected integer

Let `S_k=sum_{i+j=k} a_i*b_j`, and let u_k be the exact incoming carry.
Then `V_k=S_k+u_k` is an integer, at most 350.

The uniform gather at p=11+k divides by p+1. Unit 31+k reverses this using
its output coefficient `(p+1)/128`, with GELU input scale 128.
Its gate is `64*(position_indicator-1) - 64*(r-5)^2`.

Ignoring calibration, GELU, and rounding errors, its assembled quantity is:

    k=0:  S_0
    k=1:  S_1 + (3/32)*(S_0-c_0)
    k>=2: S_k + (51/512)*(S_{k-1}-c_{k-1})
                   + (5/512)*(S_{k-2}-c_{k-2})

Nonexistent columns are zero, including S_7.

The ideal uncoarsened two-prior-column expression would be

    V_k - u_{k-2}/100   (k>=2).

For k=1, replacing 1/10 by 3/32 underestimates by at most 0.5.
For k>=2, the extra coefficient errors are

    1/10 - 51/512 = 1/2560,
    1/100 - 5/512 = 3/12800.

Using column maxima `[81,162,243,324,243,162,81]` and incoming carry maxima
`[0,8,17,26,35,27,18,9]`, the largest combined ideal underestimate for k>=2
is at most 0.470235. Small overestimates are possible when a signed lower
column is negative; this adds at most about 0.006 before numerical errors.

The sums of the scaled assembly coefficients are bf16-exact:
128, 140, 142, or 14 as appropriate. One reference weight therefore cancels
the mean for every assembly unit. The second reference coordinate is removed.

Near-negative values for V_k=0 may be clipped toward zero by the assembly
GELU, which remains compatible with digit zero.

### Passes 7..41: modulo ten

The two units apply

    x <- x - (40/64) *
             [g(64*(x-9.125)) - g(64*(x-9.375))].

In the ReLU limit this subtracts ten above 9.375 and zero below 9.125.
A value representing integer 9 stays below the lower threshold; a value
representing integer 10 with the bounded carry underestimate stays above
the upper threshold. Thirty-five subtractions suffice for V<=350.

Input scale 64 is important. A scale-32 version produced small GELU negative-tail
increments near 9.01; these accumulated across loops and caused 3 failures
in 16,000 problems. Doubling input scale and halving output weights removed
that failure mechanism without changing the ideal map or parameter count.

Before assembly, total is zero, so the modulo units are inactive.
The readout pass follows the completed reduction.

### Pass 42: tied quadratic readout with strict marker separation

The readout gate uses `255*(r-41)`. Earlier passes are suppressed.
The two readout units use `total+1.75` and `d+1`, leaving approximately

    coordinate 0 = total+0.75,
    coordinate 1 = 1,
    coordinate 2 = original T(d), ignored by final LN gain.

The first two coordinates' difference is `total-0.25`.
Their equal final LN gains cancel the common mean.

For output digit j, the logit is approximately

    4 * [j*(total+0.25) - j^2/2] + 1.25.

Thus the digit is selected by nearest-integer classification of total+0.25.
The precise formula replaces the factor 4 on the coordinate difference by
`2^24/sigma`; the anchor pair keeps this very close to 4.

The triangular penalty comes from final LN bias -4, not from clearing a
triangle residual coordinate. This removes the old triangle-clearing neuron.

Finally, the coordinate-20 bias `2^-24` dotted with the existing signed
anchor gives every digit +1.25 and every marker -1.25. In particular, digit
zero has logit +1.25, strictly above all four markers. No tie-breaking is
used for acceptance.

## Nonzero accounting

The tied token/output embedding is counted only once.

| Tensor | Nonzero entries |
|---|---:|
| `wte` | 54 |
| `wpe` | 19 |
| `lnf.w` | 2 |
| `lnf.b` | 2 |
| `h0.ln1.w` | 12 |
| `h0.ln1.b` | 0 |
| `h0.attn.wqkv` | 22 |
| `h0.attn.bqkv` | 2 |
| `h0.attn.wo` | 8 |
| `h0.attn.bo` | 0 |
| `h0.ln2.w` | 30 |
| `h0.ln2.b` | 15 |
| `h0.mlp.w1` | 141 |
| `h0.mlp.b1` | 5 |
| `h0.mlp.w2` | 49 |
| `h0.mlp.b2` | 3 |
| **Total** | **364** |

The embedding has 26 digit-code entries and 28 anchor entries.
Position embeddings contain 12 gate indicators and 7 address entries.
The largest remaining cost is the MLP input matrix, chiefly the 23 routing
units, eight assembly units, and explicit mean rejection.

### Reduction history

| Change | Nonzeros |
|---|---:|
| Supplied earlier model | 438 |
| Constant triangle readout | 432 |
| Balanced routing gates | 409 |
| Merge quadratic key and validity flag | 402 |
| Strict marker separation | 403 |
| Factor position-minus-one LN biases | 384 |
| Shared squared-clock update | 373 |
| Difference-form GELU multiplier | 372 |
| Coarse carry coefficients and one reference | 364 |

The old 438-nonzero model and early argmax-only checks do not satisfy the
later strict requirement. All claimed final acceptance results use the
supplied strict checker against all 14 tokens.

A separate 358-nonzero single-anchor experiment was rejected: it failed all
8,000 strict test problems. Large centered values caused small clock errors
which the routing/assembly gates amplified. It is not in the delivered weights.

## Bfloat16 robustness and replay

All stored weights are exactly bf16-representable: powers of two, small
integers, and explicitly chosen dyadic fractions. Reference sums were chosen
or decomposed to retain that property. The squared-clock, readout, and
modulo changes also preserve it.

Weight rounding is therefore exactly identity, verified entry by entry.
Float32 arithmetic error, GELU approximation, and weak signal dependence of
the LayerNorm scale remain relevant; the measured strict margins assess
their combined effect. Two independent final-model random tests and the
boundary sweep provide evidence, not an exhaustive numerical proof.

The builder writes config.json and all sixteen full float32 .npy tensors,
including zero tensors. Its fresh-process output was compared with both the
in-memory network and the installed weights.

## What I think the true minimum is

364 is a demonstrated upper bound, not a proven minimum. I have no useful
architecture-wide lower bound. My tentative guess is that a more economical
serial arithmetic representation could get to roughly 300 nonzeros, but that
is speculation, not an experimentally established target or bound.

The reasons I suspect room below 364 are the 141 MLP input entries, the
23 separate product/answer routing gates, and duplicated timing/address
infrastructure. The reasons I do not claim a much smaller construction are
the costs of causal access, robust decimal carry, tied readout, and LayerNorm
mean rejection under bf16 weight constraints. Sharing more work across the
allowed 64 passes may save arithmetic wiring but can add clock and routing
overhead. The failed single-anchor reduction illustrates why eliminating
seemingly redundant stabilization parameters is not automatically safe.
