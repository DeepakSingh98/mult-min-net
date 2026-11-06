"""The any-length task, the network and the weight-file format (length-free variant of mult-min-net's arch.py).

TASK. n-digit x m-digit multiplication for any n, m >= 1. Every number is written least-significant digit first, one
token per digit:

    BOS a0 a1 .. a(n-1) * b0 .. b(m-1) EOS c0 c1 .. c(n+m-1)

c = a * b zero-padded to n + m digits. BOS takes the old H1 slot (token 12); the old H2 slot (13) is BUSY, an output
only token that never appears in the input. The model predicts c_k from position n + m + 2 + k (the EOS or the
previous answer digit), i.e. logits[:, n+m+2+k] predict tokens[:, n+m+3+k]. Scoring is teacher-forced; for a network
that is right at every digit this equals greedy decoding (attention is causal, so position p never sees tokens after p).

NETWORK. GPT-2-style pre-LayerNorm blocks with gelu_new MLPs, LayerNorm eps 1e-5 and an output layer tied to the
input embedding, exactly as in mult-min-net, with two length-free changes:
  1. no position-embedding table. The only position signal is an ALiBi-style linear distance bias: head h of block l
     adds -alibi[l][h] * (t - s) to the score of query t on key s <= t (Press et al. 2022).
  2. the stack of n_layer blocks is applied repeatedly (a looped block, as in mult-min-net's looped network). The
     network decides when it is done: HALTING RULE (outside the network, like the adaptive looped nets): after each
     pass apply the final LayerNorm + tied readout and stop at the first pass at which no position at or after EOS
     has BUSY as its argmax; the answer is read at that pass. forward(..., K=k) instead runs exactly k passes.
Config keys: n_layer, n_head, d_head, d_model, d_mlp, alibi (n_layer lists of n_head slopes).
Weights are a flat dict of float32 tensors; linear maps are x @ W + b; wqkv's columns are q, k, v (each
n_head * d_head wide) and each of them is split into heads in order.

FILES. *.npz: uncompressed zip of .npy arrays plus "config" (a JSON string); save() is byte-for-byte deterministic.
"""
import io
import json
import math
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

VOCAB, EPS = 14, 1e-5
MUL, EOS, BOS, BUSY = 10, 11, 12, 13
TOKEN_NAMES = [str(i) for i in range(10)] + ["*", "EOS", "BOS", "BUSY"]
CFG_KEYS = ("n_layer", "n_head", "d_head", "d_model", "d_mlp", "alibi")


def shapes(cfg):
    D, A, M = cfg["d_model"], cfg["n_head"] * cfg["d_head"], cfg["d_mlp"]
    S = {"wte": (VOCAB, D), "lnf.w": (D,), "lnf.b": (D,)}
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


def readout(W, x):
    """final LayerNorm + tied output layer."""
    return F.layer_norm(x, (x.shape[-1],), W["lnf.w"], W["lnf.b"], EPS) @ W["wte"].T


def forward(W, tokens, cfg, K=None, K_max=100000, every=None):
    """K given: logits [N, T, VOCAB] after exactly K passes. K None: halting rule; returns (logits taken at each row's
    halting pass, halting passes [N]); raises if some row is still busy after K_max passes. every(r, x) is called
    after each pass r = 1, 2, ... (diagnostics)."""
    N, T = tokens.shape
    D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
    A = H * DH
    x = W["wte"][tokens]
    pos = torch.arange(T, device=x.device)
    dist = (pos[:, None] - pos[None, :]).to(x.dtype)                     # t - s
    bias = []
    for l in range(cfg["n_layer"]):
        sl = torch.tensor(cfg["alibi"][l], dtype=x.dtype, device=x.device)
        bias.append((-sl[:, None, None] * dist).masked_fill(dist < 0, float("-inf")))   # [H, T, T]
    after_eos = pos[None, :] >= (tokens == EOS).int().argmax(1)[:, None]  # [N, T]
    halt = torch.zeros(N, dtype=torch.long)
    out = None
    r = 0
    while True:
        r += 1
        for l in range(cfg["n_layer"]):
            h = F.layer_norm(x, (D,), W[f"h{l}.ln1.w"], W[f"h{l}.ln1.b"], EPS)
            qkv = h @ W[f"h{l}.attn.wqkv"] + W[f"h{l}.attn.bqkv"]
            q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
            s = (q @ k.transpose(-1, -2)) / math.sqrt(DH) + bias[l]
            z = s.softmax(-1) @ v
            x = x + z.transpose(1, 2).reshape(N, T, A) @ W[f"h{l}.attn.wo"] + W[f"h{l}.attn.bo"]
            h = F.layer_norm(x, (D,), W[f"h{l}.ln2.w"], W[f"h{l}.ln2.b"], EPS)
            x = x + gelu_new(h @ W[f"h{l}.mlp.w1"] + W[f"h{l}.mlp.b1"]) @ W[f"h{l}.mlp.w2"] + W[f"h{l}.mlp.b2"]
        if every is not None:
            every(r, x)
        if K is not None:
            if r == K:
                return readout(W, x)
            continue
        lg = readout(W, x)
        done = ~((lg.argmax(-1) == BUSY) & after_eos).any(1) & (halt == 0)
        if out is None:
            out = torch.zeros_like(lg)
        out[done] = lg[done]
        halt[done] = r
        if (halt > 0).all():
            return out, halt
        if r >= K_max:
            raise RuntimeError(f"still busy after {K_max} passes")


def nonzero(W):
    return int(sum(int((v != 0).sum()) for v in W.values()))


def dense(cfg):
    return int(sum(int(np.prod(s)) for s in shapes(cfg).values()))


def to_bf16(W):
    return {k: v.bfloat16().float() for k, v in W.items()}


def encode(A, B):
    """token sequences [N, n+m+3+n+m] for digit arrays A [N, n], B [N, m] (int, LSB first); returns (tokens, C) with
    C [N, n+m] the product's digits, LSB first. Python ints carry the exact product at any length."""
    A, B = np.asarray(A, dtype=np.int64), np.asarray(B, dtype=np.int64)
    N, n = A.shape
    m = B.shape[1]
    C = np.zeros((N, n + m), dtype=np.int64)
    for r in range(N):
        a = int("".join(map(str, A[r, ::-1])))
        b = int("".join(map(str, B[r, ::-1])))
        s = str(a * b).zfill(n + m)[::-1]
        C[r] = [int(ch) for ch in s]
    t = np.concatenate([np.full((N, 1), BOS), A, np.full((N, 1), MUL), B, np.full((N, 1), EOS), C], axis=1)
    return torch.from_numpy(t), C


def margins(W, A, B, cfg):
    """per problem, at the pass where the halting rule stops: (all n+m digits right under argmax [N] bool, strict
    margin [N] = min over answer digits of correct logit - best other logit (strictly right iff > 0, ties wrong),
    per-digit margins [N, n+m], halting pass [N])."""
    t, _ = encode(A, B)
    n, m = A.shape[1], B.shape[1]
    p0 = n + m + 2
    lg, halt = forward(W, t, cfg)
    lg = lg[:, p0:p0 + n + m]
    tg = t[:, p0 + 1:p0 + 1 + n + m]
    ok = (lg.argmax(-1) == tg).all(1)
    corr = lg.gather(-1, tg[..., None])[..., 0]
    other = lg.scatter(-1, tg[..., None], float("-inf")).max(-1).values
    dm = corr - other
    return ok, dm.min(1).values, dm, halt


def save(path, cfg, W):
    items = [("config", np.array(json.dumps(cfg, sort_keys=True)))]
    items += [(k, np.ascontiguousarray(W[k], dtype=np.float32)) for k in sorted(W)]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, arr in items:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            z.writestr(zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


def load(path):
    with np.load(path, allow_pickle=False) as f:
        cfg = json.loads(str(f["config"]))
        W = {k: torch.from_numpy(f[k].astype(np.float32)) for k in f.files if k != "config"}
    if set(W) != set(shapes(cfg)) or any(tuple(W[k].shape) != s for k, s in shapes(cfg).items()):
        raise ValueError(f"{path}: tensors do not match the config's shapes")
    return cfg, W
