"""The task, the network and the weight-file format.

TASK. 4-digit x 4-digit multiplication: every number is written least-significant digit first, one token per digit, no
separator tokens between digits:

    position:  0  1  2  3  4  5  6  7  8  9   10  11  12 13 ... 19
    token:     a0 a1 a2 a3 *  b0 b1 b2 b3 EOS H1  H2  c0 c1 ... c7

a = a0 + 10 a1 + 100 a2 + 1000 a3 and b likewise (both 1000..9999), c = a * b = c0 + 10 c1 + ... + 10^7 c7 (zero-padded
to 8 digits). EOS H1 H2 are fixed marker tokens. The model reads the sequence left to right (causal attention) and
predicts c_k from position 11 + k, i.e. logits[:, 11 + k] predict tokens[:, 12 + k]. Scoring is teacher-forced: the
earlier answer digits are visible as input. For a network that gets every digit right this equals greedy decoding.

NETWORK. A GPT-2-style decoder with a 14-token vocabulary, 20 learned positions, gelu_new MLPs, LayerNorm eps 1e-5 and
an output layer tied to the input embedding. The sizes are a config dict:
    n_layer   number of distinct blocks
    loop      K: the stack of n_layer blocks is applied K times in a row with the same weights (K = 1: no looping)
    n_head    attention heads per block;  d_head  size of each head (n_head * d_head need not equal d_model)
    d_model   residual width;  d_mlp  MLP width
Forward pass (pre-LayerNorm residual blocks):
    x = wte[tokens] + wpe[positions]
    repeat K times:
        for l in 0 .. n_layer-1:
            x = x + attn_l(ln1_l(x))     # causal; per head softmax(q k^T / sqrt(d_head)) v, heads concatenated, @ wo + bo
            x = x + mlp_l(ln2_l(x))      # gelu_new(. @ w1 + b1) @ w2 + b2
    logits = lnf(x) @ wte^T
Weights are a flat dict of float32 tensors (names from shapes(cfg)); linear maps are x @ W + b. The qkv projection
h @ wqkv + bqkv is split into q, k, v (each n_head * d_head wide, in that order), and each of them into heads in order.

FILES. weights/*.npz is an uncompressed zip of .npy arrays (np.load reads it; no pickle): one dense float32 array per
tensor plus "config", a JSON string. save() writes it byte-for-byte deterministically, so its sha256 identifies the
weights.
"""
import io
import json
import math
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

VOCAB, NPOS, EPS = 14, 20, 1e-5
MUL, EOS, H1, H2 = 10, 11, 12, 13
TOKEN_NAMES = [str(i) for i in range(10)] + ["*", "EOS", "H1", "H2"]
ANS0 = 12             # tokens[:, 12:20] are c0..c7
PRED0 = 11            # logits[:, 11:19] predict them
CFG_KEYS = ("n_layer", "loop", "n_head", "d_head", "d_model", "d_mlp")


def shapes(cfg):
    """dict tensor name -> shape for config cfg."""
    D, A, M = cfg["d_model"], cfg["n_head"] * cfg["d_head"], cfg["d_mlp"]
    S = {"wte": (VOCAB, D), "wpe": (NPOS, D), "lnf.w": (D,), "lnf.b": (D,)}
    for l in range(cfg["n_layer"]):
        S.update({f"h{l}.ln1.w": (D,), f"h{l}.ln1.b": (D,),
                  f"h{l}.attn.wqkv": (D, 3 * A), f"h{l}.attn.bqkv": (3 * A,),
                  f"h{l}.attn.wo": (A, D), f"h{l}.attn.bo": (D,),
                  f"h{l}.ln2.w": (D,), f"h{l}.ln2.b": (D,),
                  f"h{l}.mlp.w1": (D, M), f"h{l}.mlp.b1": (M,),
                  f"h{l}.mlp.w2": (M, D), f"h{l}.mlp.b2": (D,)})
    return S


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def forward(W, tokens, cfg):
    """logits [N, T, VOCAB] for int64 tokens [N, T]."""
    N, T = tokens.shape
    D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
    A = H * DH
    x = W["wte"][tokens] + W["wpe"][:T]
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    for _ in range(cfg["loop"]):
        for l in range(cfg["n_layer"]):
            h = F.layer_norm(x, (D,), W[f"h{l}.ln1.w"], W[f"h{l}.ln1.b"], EPS)
            qkv = h @ W[f"h{l}.attn.wqkv"] + W[f"h{l}.attn.bqkv"]
            q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
            s = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
            a = s.masked_fill(mask, float("-inf")).softmax(-1)
            z = a @ v                                                    # [N, H, T, DH]
            o = z.transpose(1, 2).reshape(N, T, A) @ W[f"h{l}.attn.wo"] + W[f"h{l}.attn.bo"]
            x = x + o
            h = F.layer_norm(x, (D,), W[f"h{l}.ln2.w"], W[f"h{l}.ln2.b"], EPS)
            m = gelu_new(h @ W[f"h{l}.mlp.w1"] + W[f"h{l}.mlp.b1"]) @ W[f"h{l}.mlp.w2"] + W[f"h{l}.mlp.b2"]
            x = x + m
    h = F.layer_norm(x, (D,), W["lnf.w"], W["lnf.b"], EPS)
    return h @ W["wte"].T


def nonzero(W):
    """the score: number of nonzero entries over all tensors (the tied embedding counts once)."""
    return int(sum(int((v != 0).sum()) for v in W.values()))


def dense(cfg):
    """size of the config: every entry, zero or not."""
    return int(sum(int(np.prod(s)) for s in shapes(cfg).values()))


def to_bf16(W):
    """every weight rounded to bfloat16 and back to float32 (the forward pass still computes in float32)."""
    return {k: v.bfloat16().float() for k, v in W.items()}


def digits(x, n):
    """least-significant digit first, [len(x), n] int64."""
    x = np.asarray(x, dtype=np.int64)
    return np.stack([(x // 10 ** i) % 10 for i in range(n)], axis=1)


def encode(a, b):
    """token sequences [N, 20] (int64 torch) for operand arrays a, b (1000..9999)."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    n = len(a)
    t = np.concatenate([digits(a, 4), np.full((n, 1), MUL), digits(b, 4), np.full((n, 1), EOS),
                        np.full((n, 1), H1), np.full((n, 1), H2), digits(a * b, 8)], axis=1)
    return torch.from_numpy(t)


def margins(W, a, b, cfg):
    """per problem: (all 8 digits right under argmax [N] bool, strict margin [N]). The strict margin is the smallest,
    over the 8 answer digits, of (correct token's logit - largest of the other 13 logits); the problem is strictly
    right iff it is > 0 (a tie counts as wrong)."""
    t = encode(a, b)
    lg = forward(W, t, cfg)[:, PRED0:PRED0 + 8]
    tg = t[:, ANS0:ANS0 + 8]
    ok = (lg.argmax(-1) == tg).all(1)
    corr = lg.gather(-1, tg[..., None])[..., 0]
    other = lg.scatter(-1, tg[..., None], float("-inf")).max(-1).values
    return ok, (corr - other).min(1).values


def save(path, cfg, W):
    """write weights/NAME.npz deterministically (sorted names, fixed timestamps, no compression)."""
    items = [("config", np.array(json.dumps(cfg, sort_keys=True)))]
    items += [(k, np.ascontiguousarray(W[k], dtype=np.float32)) for k in sorted(W)]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, arr in items:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            z.writestr(zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


def load(path):
    """(cfg, W) with W a dict of float32 torch tensors."""
    with np.load(path, allow_pickle=False) as f:
        cfg = json.loads(str(f["config"]))
        W = {k: torch.from_numpy(f[k].astype(np.float32)) for k in f.files if k != "config"}
    if set(W) != set(shapes(cfg)) or any(tuple(W[k].shape) != s for k, s in shapes(cfg).items()):
        raise ValueError(f"{path}: tensors do not match the config's shapes")
    return cfg, W
