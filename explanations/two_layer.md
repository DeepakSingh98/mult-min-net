# A 487-nonzero, two-block multiplication transformer

> Astra's own write-up of this network, lightly edited for this repository: its in-session test tables (seeds it
> chose itself) are replaced by a pointer to the README, whose results were measured separately after delivery;
> sandbox paths are replaced by repository paths; "parameters" is changed to "nonzero weights" where it counts
> nonzeros. "The supplied previous/earlier build" means Astra's own earlier, larger build for this arm (not shipped
> here). Everything else, including its guesses about the minimum, is Astra's text.

## Delivered network

Configuration: `n_layer=2, loop=1, n_head=8, d_head=2, d_model=512, d_mlp=69`.
There are **487 nonzero weights; the dense size is 231,658**.
Both blocks are distinct. The tied embedding is counted once.

The builder (`build/two_layer.py`) initializes every full tensor to float32 zero and writes its explicit constants.
Its only dependency is `arch.py`; it does not read earlier builders, examples, or saved weights.
Measured strict results are in the README.

No training, gradients, optimization, data fitting, or automated weight search
was used. All coefficients are algebraic constants or explicit BF16 rounding
of constants. The construction reuses and modifies the supplied previous
511-nonzero program.

## Embeddings and normalization

Residual coordinates:

| Coordinates | Meaning |
|---|---|
| 0, 1 | Input `(d,-d)`; later the readout pair |
| 2 | `d(d-1)`, the tied quadratic code |
| 4, 5 | Balanced anchors `(+R,-R)`, R=2^18 |
| 6, 7 | Position z=p-4 and z^2 |
| 8..11 | Retrieved a digits |
| 12..15 | Retrieved shifted b digits |
| 16..19 | Shifted-source query codes |
| 20 | Scaled convolution column |
| 23 | Negative-sum balancing coordinate |

Each digit also puts `-d(d-1)` in coordinate 23. All four marker embeddings
are identically zero. Fixed positions 9, 10, and 11 carry the anchor pair
instead, so EOS/H1/H2 have the same input residuals as anchored markers would.
Position 4 is an exactly zero dummy source. Position 19 has zero positional
embedding because it cannot influence any scored prediction.

For p<19, position embeddings supply z, z^2, and query i equal to j+1 when
j=p-11-i is in 0..3, or zero otherwise. Coordinate 23 balances the entire row.
Attention and first-MLP outputs are also balanced.

The nominal standard deviation is sigma=R/16=2^14. Active LayerNorm gains
are sigma, restoring arithmetic coordinates approximately after normalization.
Inactive gains are zero. The large anchor makes variance changes small; zero
padding reduces centering errors. This is approximate internal arithmetic,
followed by exact discrete decisions, not exact multiplication in real GELU
arithmetic.

## Block 0 attention: eight digit reads

At prediction position p=11+k:

* Heads i=0..3 retrieve a_i from fixed source i.
* Heads 4+i retrieve b_(k-i), or the zero dummy when that index is invalid.

Every head has q=(128*t,-64), key=(z,z^2), and digit in value channel 0.
Its score is

    (64/sqrt(2)) * (2*t*z-z^2).

The target t is i-4 for a heads, or the appropriate query coordinate for b
heads. A neighboring source is down by about 45.25 logits in nominal
arithmetic. Each head writes the retrieved digit to its operand coordinate
and its negative to coordinate 23.

## Block 0 MLP: nine GELUs for the convolution

Let g be the specified gelu_new, e=2^-10, C=2^20, and
c=1/sqrt(2*pi). Near zero,

    g(x) = x/2 + c*x^2 + O(x^4).

For each retrieved pair (a_i,b_i), two neurons contribute

    C * [g(e*(a_i+b_i)) - g(e*(a_i-b_i))].

Their linear term is C*e*b_i, and their leading quadratic term is
4*c*e^2*C*a_i*b_i. A ninth neuron subtracts

    (C*e/16) * g(16*sum_i b_i).

The b sum is nonnegative. At positive integer sums, tanh is saturated and
g(16*s)=16*s in float32; g(0)=0. Soft-selection and normalization perturbations
near zero are small and covered by the tests.

Thus coordinate 20 approximately contains

    P_k = sqrt(8/pi)*S_k,
    S_k = sum_(i+j=k) a_i*b_j,

and coordinate 23 receives its negative. Product-unit arguments are at most
18/1024. The leading GELU truncation error in a four-product unscaled column
is about 0.017 or less. Floating-point and normalization errors are additional.

## Block 1 attention: approximate carry reconstruction

Heads h=0,1,2 read positions p-h using the same quadratic selection mechanism;
all other heads are zero.

Let q_k be the carry entering column k, N_k=S_k+q_k, and c_k=N_k mod 10.
In nominal arithmetic the heads form

    T_k = S_k + (S_(k-1)-c_(k-1))/10
              + (S_(k-2)-c_(k-2))/100
        = N_k - q_(k-2)/100.

Negative-index quantities vanish. Carries are at most 35, so omitted carry
is in [0,0.35]. The final readout centers it with +3/16.

Each head's column coefficient approximates f_h=sqrt(pi/8)*10^-h using
a product, not a sum of channels:

    step = 2^(floor(log2(f_h))-7)
    outer = ceil(f_h/step)*step
    inner = BF16_nearest(f_h/outer).

The value projection multiplies coordinate 20 by inner; the output projection
writes outer times that value to coordinate 0 and its negative to coordinate 1.
Both factors are BF16-exact. For all three heads inner=255/256.

| h | outer | effective error per unscaled S column |
|---|---:|---:|
| 0 | 0.62890625 | -0.0003310961 |
| 1 | 0.06298828125 | +0.0001221185 |
| 2 | 0.00628662109375 | -0.0000071917 |

Head 0 also reads the current token digit with coefficient -1.1, combining
residual-digit cancellation and -0.1*c_(k-1). Head 1 reads the preceding digit
with coefficient -0.01. These contributions are balanced pairs. Consequently
the readout residual is approximately `(T,-T)` with zero total residual mean.

## Block 1 MLP: 34 resets plus one top-case hinge

LayerNorm biases at coordinates 0 and 1 are respectively 15/16 and -7/16.
For m=1..34, use two neurons:

    u_m = g(64*(T+15/16-10*m))
    v_m = g(64*(T+ 7/16-10*m)).

Output coefficients -40/64 and +40/64 write only to coordinate 0.
Above a transition, u_m-v_m is approximately 32, subtracting 20 from the
readout difference. Below it both are approximately zero. The half-unit
separation is encoded in small LayerNorm biases, not large threshold biases,
which protects it from BF16 rounding.

The largest N is 350, attained only in column 3 of 9999*9999. Indeed, S_3<=324,
q_3<=26, and any non-nine operand digit reduces S_3 to at most 315, giving
N_3<=341. All other columns have maxima at most 278. Thus there is no other
case above 341.

One final neuron uses

    g(64*(T+15/16-344)) * (-4/64).

It is inactive for all ordinary cases and pushes the isolated N=350 case
well into digit zero's decision region. It need not reproduce the exact
numeric remainder: digit zero wins throughout the lower half-line.
This replaces the last two-neuron reset, leaving 69 neurons.

## Strict tied readout

Final LayerNorm has gains sigma on coordinates 0 and 1, biases -5/8 on
coordinate 0 and -1 on coordinate 2, and bias 1/R on coordinate 4.
All other final gains and biases are zero.

For ordinary cases, with m crossed resets, the digit-d logit is approximately

    d*(2*T-20*m-5/8) - d*(d-1) + 1
      = 2*d*r - d^2 + 1,
    r = T + 3/16 - 10*m.

It chooses the nearest digit. The isolated top case instead has r safely
below 1/2, also selecting zero.

Every digit receives +1 through its anchor embedding and the final 1/R bias.
Every marker embedding is zero, giving marker logits exactly zero. Digit
zero's logit is exactly +1, so there are no marker ties. The shared +1 does
not alter competition between digits. Final LayerNorm centering cancels in
the difference of coordinates 0 and 1.

## Why BF16 survives

Only four entries change under BF16 rounding: the paired output coefficients
-1.1 and -0.01. Their rounded magnitudes are 1.1015625 and 0.010009765625.
The direct perturbation is bounded by

    9*(0.0015625 + 0.000009765625) < 0.014151.

All other weights are BF16-exact: integer embedding codes, power-of-two
anchors and gains, dyadic biases and outputs, thresholds -640*m and -64*344,
and the explicitly rounded coefficient factors. Their product approximation
errors exist in both precision modes; they are not additional BF16 errors.

Centering omitted carry gives nominal error in [-0.1625,0.1875].
Coefficient, GELU, normalization, and reduction errors consume some remaining
decision margin. The strict tests measure the combined effect rather than
claiming a hardware-independent exhaustive error proof.

## Nonzero accounting

All unlisted entries within each tensor are exactly zero.

| Tensor | Nonzeros | Dense entries |
|---|---:|---:|
| `wte` | 54 | 7,168 |
| `wpe` | 75 | 10,240 |
| `lnf.w` | 2 | 512 |
| `lnf.b` | 3 | 512 |
| `h0.ln1.w` | 7 | 512 |
| `h0.ln1.b` | 0 | 512 |
| `h0.attn.wqkv` | 28 | 24,576 |
| `h0.attn.bqkv` | 12 | 48 |
| `h0.attn.wo` | 16 | 8,192 |
| `h0.attn.bo` | 0 | 512 |
| `h0.ln2.w` | 8 | 512 |
| `h0.ln2.b` | 0 | 512 |
| `h0.mlp.w1` | 20 | 35,328 |
| `h0.mlp.b1` | 0 | 69 |
| `h0.mlp.w2` | 18 | 35,328 |
| `h0.mlp.b2` | 0 | 512 |
| `h1.ln1.w` | 4 | 512 |
| `h1.ln1.b` | 0 | 512 |
| `h1.attn.wqkv` | 14 | 24,576 |
| `h1.attn.bqkv` | 5 | 48 |
| `h1.attn.wo` | 10 | 8,192 |
| `h1.attn.bo` | 0 | 512 |
| `h1.ln2.w` | 2 | 512 |
| `h1.ln2.b` | 2 | 512 |
| `h1.mlp.w1` | 69 | 35,328 |
| `h1.mlp.b1` | 69 | 69 |
| `h1.mlp.w2` | 69 | 35,328 |
| `h1.mlp.b2` | 0 | 512 |
| **Total** | **487** | **231,658** |

Embeddings: 129; block 0: 109; block 1: 244; final LayerNorm: 5.
The second MLP alone costs 207 entries.

Relative to the supplied 511-entry design: positive-sum cancellation saves 6;
merged digit channel saves 3; d(d-1) saves 2; moving offsets saves 2;
multiplicative coefficient encoding saves 9; isolated top hinge saves 3.
The strict-margin fix costs 1. Net saving: 24.

## Minimum and remaining uncertainty

487 is a demonstrated upper bound, not a proven minimum. I suspect a different
modulo decoder or more economical routing could lower it, perhaps into the
350–480 range, but this is conjecture. The 207-entry decoder and 129-entry
embeddings are the main opportunities. I have no nontrivial global lower
bound for this architecture and do not claim optimality.

The principal remaining uncertainty is exhaustive numerical correctness on
untested pairs or different float32 reduction implementations. The two
485-entry residual-cancellation experiments failed and were rejected; they
are not part of the delivered builder. The final submission is the verified
487-entry strict-margin construction.
