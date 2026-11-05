# Hand-built one-layer multiplier: 927 nonzero weights

> Astra's own write-up of this network, lightly edited for this repository: its in-session test tables (seeds it
> chose itself) are replaced by a pointer to the README, whose results were measured separately after delivery;
> sandbox paths are replaced by repository paths; "parameters" is changed to "nonzero weights" where it counts
> nonzeros. "The supplied previous/earlier build" means Astra's own earlier, larger build for this arm (not shipped
> here). Everything else, including its guesses about the minimum, is Astra's text.

## Delivered network

Configuration: `n_layer=1, loop=1, n_head=24, d_head=4, d_model=96, d_mlp=161`.

**927 nonzero weights; the dense size is 72,257.** This is 161 fewer
nonzeros than the supplied 1,088-nonzero construction, while satisfying
the strict-margin requirement (the correct token's logit strictly above all
13 others). There is exactly one attention layer, one GELU MLP, and the tied
readout. Measured strict results are in the README.

`python build/one_layer.py OUTDIR` constructs all tensors from scratch. The
builder reads only `arch.py`; it does not read previous builds, test data, or
notes.

No training, gradients, optimization, statistical fitting, least squares,
or automated weight/mask search was used. The exact integer padding
constants solve sum and norm equations, not equations fitted to examples.
They were obtained by exact integer norm completion and are written
directly into the standalone builder, with algebraic assertions.

## 1. Origin and reductions

This construction derives from the supplied previous build. Its
softmax-curvature multiplication identity, prefix subtraction, separable
modular correction, and sawtooth decoder are retained.

Changes made in this session:

* Merge 16 uniform pair heads into seven diagonal-average heads: 1,088 to 980.
* Absorb redundant token features into norm padding and center each MLP
  group at a hinge: 980 to 958.
* Share, then eliminate, negative tied-readout feature coordinates:
  958 to 948 to 937.
* Move the MLP constant from tokens into prediction-position embeddings:
  937 to 926.
* Add one digit-only final bias for strict separation from markers: 927.

Earlier argmax-only checks are not evidence of strict acceptance.
The final strict tests above were all performed on the 927-nonzero model.

## 2. Embeddings and first LayerNorm

Let f=floor(d/2). A digit token has:

    e[0] = d/2
    e[2] = d^2/8
    e[4] = f/2
    e[6:10] = PADDING[d]/8

All other entries are zero. All four marker embeddings are entirely zero.
The integer padding rows are:

    d=0: [-128,-128,128,128]
    d=1: [-219,35,102,77]
    d=2: [-224,52,92,64]
    d=3: [-227,57,74,71]
    d=4: [-228,36,84,68]
    d=5: [-231,49,78,51]
    d=6: [-232,28,76,56]
    d=7: [-231,11,74,57]
    d=8: [-232,8,56,56]
    d=9: [-231,14,55,29]

For each row P, the builder verifies:

    sum(P) = -(4d+d^2+4f)
    sum(P_i^2) = 65536-16d^2-d^4-16f^2.

Consequently every digit embedding has mean zero and squared norm 1024.
All these entries are exactly representable in bfloat16.

Operand position p in {0,1,2,3,5,6,7,8} has +16 at coordinate
u_p=16+2p and -16 at u_p+1. Prediction position p=11+k has -16 at
v_k=56+2k and +16 at coordinate 14. All other position entries are zero.
Each nonzero position vector has zero sum and squared norm 512.

Selected operand and history digits therefore have pre-LayerNorm variance
1536/96=16. First-LayerNorm gain 4 is enabled only on digit features
0,2,4, the eight operand selector coordinates, and the seven history
selector coordinates v_1,...,v_7. It is nearly the identity there; EPS
introduces a common relative scale perturbation of about 3.1e-7.
All queries are biases, so H2 need not have a digit's norm.

## 3. Attention: seven uniform heads, sixteen nonlinear heads, one prefix head

All heads have width four. Except for the prefix head's second score
component, q[0]=2 and other query entries are zero. This cancels the
attention divisor sqrt(4)=2.

### Uniform diagonal heads 0 through 6

For diagonal s, define

    I_s = {(i,j): 0<=i,j<4, i+j=s},   m_s=|I_s|.

Head s selects the 2*m_s operand positions appearing in these pairs.
Their key score is 32; unselected scores are zero. Its two value
components on a selected digit d are

    m_s*(1024*d+d^2),   m_s*floor(d/2).

Thus its outputs are, up to tiny routing/normalization errors,

    U_s = sum_{(i,j) in I_s} [
                1024*(a_i+b_j)/2 + (a_i^2+b_j^2)/2 ]
    V_s = sum_{(i,j) in I_s} [
                (floor(a_i/2)+floor(b_j/2))/2 ].

This is exactly the sum of the previous uniform pair-head outputs.
Multiplying the values by m_s, rather than combining decimal output
weights, avoids an additional source of bfloat16 cancellation error.
The value weights 2048*m_s, 8*m_s, 2*m_s are all BF16-exact.

### Nonlinear pair heads 7 through 22

Head 7+4i+j selects positions i and 5+j. It adds d/512 to each selected
digit's key score and uses value d. For a selected pair x,y, its ideal output is

    Z(x,y) = (x*exp(x/512)+y*exp(y/512)) /
             (exp(x/512)+exp(y/512))
           = (x+y)/2 + (x-y)/2*tanh((x-y)/1024).

The combination

    P(x,y) = (x^2+y^2)/2 - 1024*(Z(x,y)-(x+y)/2)

approximates xy. Its leading error is (x-y)^4/(6*1024^2), at most about
0.001043 for decimal digits. Float32 cancellation and routing errors
are additional and are included in the measured tests.

The seven uniform heads supply the summed square and linear terms,
and the sixteen biased heads supply the sixteen nonlinear terms.
Attention therefore performs nonlinear multiplication, rather than
merely retrieving operands for an enormous lookup MLP. The mechanism
is softmax curvature; it does not assume the output query already
contains an operand or claim direct query-key bilinear multiplication.

### Prefix head 23

Two query components equal 2. History position 12+j has score
32+j*log(10), using separate base-routing and logarithmic components.
Because its positional selector is -16, the corresponding key weights
are -2 and -j*log(10)/16. The value is the digit itself.

At prediction position 11+k, causality permits only j<k. Ideally the output is

    Zprefix_k = sum_{j<k} 10^j*c_j / sum_{j<k} 10^j.

Coordinate k=0 does not use this head.

## 4. Arithmetic output projection, modular reduction, and position gating

Let

    q_k = sum_{i+j<=k} a_i*b_j*10^(i+j-k)
          - sum_{j<k} c_j*10^(j-k).

With correct lower answer digits this is an integer and q_k mod 10=c_k.
Prefix subtraction removes potentially arbitrarily small distances
from decimal digit boundaries.

The current diagonal uses a separable modular correction:

    r(x,y)=xy-10*floor(x/2)-10*floor(y/2)+40.

For decimal x,y, r is an integer in [0,41], congruent to xy modulo ten.
Define

    R_k=q_k-10*sum_{i+j=k}(floor(a_i/2)+floor(b_j/2))+40*m_k.

Use m=[1,2,3,4,3,2,1,0]. Valid bounds on R_k are

    0 <= R_k <= U_k,  U=[41,90,140,190,158,109,59,9].

For each diagonal s<=k, the output coefficient on U_s is
-10^(s-k)/64. Each nonlinear pair head on that diagonal has coefficient
+1024*10^(s-k)/64. V_s contributes +20/64 only to coordinate v_s.
The prefix coefficient for coordinate v_k, k>=1, is
+(1-10^(-k))/(9*64).

Use centers C=[20,40,60,80,60,40,20,0].
The attention bias at v_k is

    16+(C_k-40*m_k)/64.

Together with the -16 prediction-position entry, the selected arithmetic
coordinate becomes approximately (C_k-R_k)/64.

For every output row, coordinate 90 receives minus the sum of its eight
arithmetic-coordinate coefficients. Its attention bias is -123. This
preserves zero residual sum before rounding, up to computation error.
After BF16 rounding the mean is small rather than identically zero.
Decimal uniform/nonlinear coefficients preserve their power-of-two
relationship under BF16 rounding, avoiding independent large errors in
their cancelled linear terms.

At a position not assigned to v_k, the -16 positional cancellation is
absent. The negative MLP input weight therefore supplies a large
off-position penalty equivalent to subtracting 1024 from the scalar,
making the other seven neuron groups inactive.

Second-LayerNorm gains are enabled only at the eight v_k coordinates
and constant coordinate 14. If the residual mean is mu, the effective
arithmetic scalar at the selected position is

    R_eff=C_k-64*(x[v_k]-mu)/(1-mu/16).

Thresholds are compared against the same normalized constant feature.
The construction does not require fixed variance after attention.

## 5. One GELU MLP: decimal sawtooth

For an integer t in [0,U_k], define

    F(t)=ReLU(t)
         -10*sum_{n=10,20,...,10*floor(U_k/10)}
                  [ReLU(t-(n-1))-ReLU(t-n)].

Then F(t)=t mod 10. Its steepest segment has slope -9.

There are eight base neurons, 152 wrap neurons (two per 76 wraps),
and one shared constant neuron: 161 total.

Set S=4096 and O=256. A position-specific hinge at threshold h has

    w1[v_k]=-64*S
    w1[14]=-S*(h-C_k)/16.

Its preactivation is approximately S*(R_k-h)/sigma, where sigma is the
actual second-LayerNorm denominator. Using the large scale makes the
fixed gelu_new act like ReLU except near the hinge. No activation is replaced.

The hinge coefficient L is 2 for the base neuron, -20 for a wrap's first
hinge, and +20 for its second. Its sole output weight is

    w2[neuron,0]=O*L*(33/64).

The shared constant neuron has w1[14]=S/16 and w2[neuron,2]=-2*O.
All MLP biases are zero.

Writing s=33/32 and A=2*O*S/sigma, the dominant output is

    x[0] += (s*A/2)*F(R_k)
    x[2] -= A.

Centers coincide with one hinge in each group, eliminating eight
constant-coordinate weights. Thus w1 has 313 nonzeros and w2 has 161.

## 6. Tied two-coordinate readout and strict marker separation

Final-LayerNorm gains are 8 at coordinates 0 and 2, zero elsewhere.
The only final bias is b[6]=-1/1024.

Let nu be the final residual mean and tau its LayerNorm denominator.
The exact tied digit-d logit is

    [4*d*(x[0]-nu)+d^2*(x[2]-nu)]/tau + beta_d,

where beta_d=-PADDING[d][0]/8192 is positive. Marker logits are exactly
zero because marker embeddings are zero.

Ignoring the small input residual, numerical errors, and beta variation,
the two active coordinates decode the value

    r(F)=((D-1)*s*F+2)/(D-2+s*F/2),  D=96, s=33/32.

This explicitly accounts for final mean subtraction instead of adding
negative readout-feature copies. Its values at digits 0 through 9 are

    0.0213,1.0577,2.0829,3.0970,4.1002,
    5.0927,6.0747,7.0463,8.0076,8.9590.

The maximum ideal displacement is 0.1001952, leaving room for arithmetic
and GELU error. The gain correction 33/32 and all resulting MLP weights
are BF16-exact.

Digit zero has logit beta_0=0.015625, exactly, independent of the input.
The other beta_d range from 0.0267334 to 0.0283203. They perturb digit
comparisons only slightly, while every marker retains logit zero.
The final tests verify strict comparisons with every digit and marker;
the construction no longer relies on lowest-index argmax tie-breaking.

## 7. Numerical robustness and limitations

Exactly preserved by BF16 weight rounding:

* digit embeddings, integer padding, and position embeddings;
* active LayerNorm gains and the digit-only final bias;
* routing coefficients, query biases, and nonlinear score scale;
* uniform-head multiplicities and floor corrections;
* arithmetic biases and all MLP thresholds/output coefficients;
* the power-of-two relation of decimal uniform/nonlinear coefficients.

Rounded non-exactly:

* decimal output coefficients and their balancing sums;
* prefix logarithmic scores and its normalization coefficients.

Both tests compute in float32, as specified; only the weights are rounded.
On an independent 2,048-problem diagnostic sample, seed 87211:

| Quantity | Float32 | BF16-rounded |
|---|---:|---:|
| Maximum attention-residual mean magnitude | 0.000004133 | 0.000233253 |
| Maximum effective-scalar distance from an integer | 0.0196991 | 0.0226898 |
| Maximum decoded-value error relative to the true digit | 0.2307987 | 0.2588577 |

The decoded-value diagnostic excludes the small digit-dependent beta
offset, whereas the strict acceptance tests include the complete logits.
These diagnostics are measurements, not universal bounds.

## 8. Nonzero accounting

The tied token embedding is counted once.

| Tensor | Shape | Nonzeros |
|---|---|---:|
| `wte` | `(14, 96)` | 66 |
| `wpe` | `(20, 96)` | 32 |
| `lnf.w` | `(96,)` | 2 |
| `lnf.b` | `(96,)` | 1 |
| `h0.ln1.w` | `(96,)` | 18 |
| `h0.ln1.b` | `(96,)` | 0 |
| `h0.attn.wqkv` | `(96, 288)` | 131 |
| `h0.attn.bqkv` | `(288,)` | 25 |
| `h0.attn.wo` | `(96, 96)` | 160 |
| `h0.attn.bo` | `(96,)` | 9 |
| `h0.ln2.w` | `(96,)` | 9 |
| `h0.ln2.b` | `(96,)` | 0 |
| `h0.mlp.w1` | `(96, 161)` | 313 |
| `h0.mlp.b1` | `(161,)` | 0 |
| `h0.mlp.w2` | `(161, 96)` | 161 |
| `h0.mlp.b2` | `(96,)` | 0 |
| **Total** | **72,257 dense entries** | **927** |

The 131 qkv weights consist of 53 for uniform heads, 64 for nonlinear
heads, and 14 for the prefix head. The 160 output-projection weights
consist of 56, 96, and 8 for these groups, including balancing writes.
All unspecified tensor entries are exactly zero.

## 9. What is the true minimum?

I do not know. **927 is a constructive upper bound, not a minimality
certificate.** My low-confidence guess is that a different hand-built
decoder could reach roughly 700–900 nonzeros; no such smaller passing
network was established here.

The current MLP alone costs 474 nonzeros. A different digit-extraction
mechanism is therefore a plausible source of further savings. This is
not evidence that the remaining attention heads or MLP neurons are
globally necessary.

A limited lower bound applies to the current separable modular reduction:
the mixed difference of xy at corners (0,0),(0,9),(9,0),(9,9) is 81
and cannot be changed by subtracting a function of x plus a function of y.
If corrected values occupy an interval of width W, that difference is
at most 2W, so W>=40.5. The integer interval [0,41] is optimal within
that restricted mechanism, not within all one-layer transformers.

The least certain claims are global minimality and correctness outside
the finite tests: there is no exhaustive floating-point proof here.
The reported strict acceptance results and replay equality are directly
measured.
