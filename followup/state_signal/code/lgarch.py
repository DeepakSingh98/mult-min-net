"""MULT-LOGDEPTH Part 2: the task, the network and the weight-file format. Encoder-style looped GPT-2 block.

TASK. n-digit x m-digit decimal multiplication. L is a power of two with n + m <= L (the least such, or any larger one).
Tokens, one row per problem, length T = 2L + S:
    position 2i     a_i (digit i of a, least significant first) or PAD when i >= n
    position 2i+1   b_i or PAD when i >= m
    positions 2L..  S copies of SCR, S = s2 L^2 + s1 L + s0 from config["scratch"] = [s2, s1, s0] (all zero if the network needs none)
Answer: digit k of a*b, zero-padded to L digits, is read as the argmax of the logits at position 2k, k = 0..L-1. Tokens hold no answer.

NETWORK. Pre-LayerNorm block, gelu_new MLP, LayerNorm eps 1e-5, output layer tied to the token embedding, learned position table wpe,
as GPT-2, except: attention is NOT causal (each token may attend to every token), and the stack of n_layer blocks is applied K times with
the same weights. Nothing else depends on K. x0 = wte[token] + wpe[position]. Linear maps are x @ W + b; wqkv's columns are q, k, v
(each n_head * d_head wide), split into heads in order. Attention score of token t on token s in a head: q_t . k_s / sqrt(d_head).
forward(W, tokens, cfg, K) runs exactly K passes. No stopping rule exists: scoring asks for the answer to be right at K and at every later pass.

FILES. weights.npz: uncompressed arrays plus "config" (JSON string with n_layer, n_head, d_head, d_model, d_mlp, n_pos, scratch).
"""
import io
import json
import math
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

VOCAB, EPS = 13, 1e-5
PAD, SCR, BUSY = 10, 11, 12            # BUSY may be output, it never appears in the input
TOKEN_NAMES = [str(i) for i in range(10)] + ["PAD", "SCR", "BUSY"]
CFG_KEYS = ("n_layer", "n_head", "d_head", "d_model", "d_mlp", "n_pos", "scratch")


def shapes(cfg):
    D, A, M = cfg["d_model"], cfg["n_head"] * cfg["d_head"], cfg["d_mlp"]
    S = {"wte": (VOCAB, D), "wpe": (cfg["n_pos"], D), "lnf.w": (D,), "lnf.b": (D,)}
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


def scratch_len(cfg, L):
    a, b, c = cfg.get("scratch", [0, 0, 0])
    return int(a * L * L + b * L + c)


def encode(A, B, L, cfg=None):
    """A [N, n], B [N, m] int digit arrays, least significant first -> tokens [N, 2L + S] (int64 tensor)."""
    A, B = np.asarray(A, dtype=np.int64), np.asarray(B, dtype=np.int64)
    N, n, m = len(A), A.shape[1], B.shape[1]
    if n + m > L or L & (L - 1):
        raise ValueError("need n + m <= L, L a power of two")
    S = scratch_len(cfg, L) if cfg else 0
    t = np.full((N, 2 * L + S), PAD, dtype=np.int64)
    t[:, 2 * L:] = SCR
    t[:, 0:2 * n:2] = A
    t[:, 1:2 * m + 1:2] = B
    return torch.from_numpy(t)


def answer(A, B, L):
    """exact digits of a*b, [N, L], least significant first (Python integers, any length)."""
    A, B = np.asarray(A), np.asarray(B)
    out = np.zeros((len(A), L), dtype=np.int64)
    for i, (a, b) in enumerate(zip(A, B)):
        c = int("".join(map(str, a[::-1].tolist())) or "0") * int("".join(map(str, b[::-1].tolist())) or "0")
        s = str(c)[::-1]
        out[i, :len(s)] = [int(ch) for ch in s]
    return out


def readout(W, x):
    return F.layer_norm(x, (x.shape[-1],), W["lnf.w"], W["lnf.b"], EPS) @ W["wte"].T


def forward(W, tokens, cfg, K, every=None, attn=None):
    """logits [N, T, VOCAB] after exactly K passes. every(r, x) is called after pass r = 1..K (diagnostics). attn(l, h, a) sees the
    attention matrix [N, H, T, T] of each block; a return value replaces it (used for lesions)."""
    N, T = tokens.shape
    D, H, DH = cfg["d_model"], cfg["n_head"], cfg["d_head"]
    A = H * DH
    x = W["wte"][tokens] + W["wpe"][: T][None]
    for r in range(1, K + 1):
        for l in range(cfg["n_layer"]):
            h = F.layer_norm(x, (D,), W[f"h{l}.ln1.w"], W[f"h{l}.ln1.b"], EPS)
            qkv = h @ W[f"h{l}.attn.wqkv"] + W[f"h{l}.attn.bqkv"]
            q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in qkv.split(A, dim=-1))
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(DH)).softmax(-1)
            if attn is not None:
                b = attn(l, h, a)
                a = a if b is None else b
            z = a @ v
            x = x + z.transpose(1, 2).reshape(N, T, A) @ W[f"h{l}.attn.wo"] + W[f"h{l}.attn.bo"]
            h = F.layer_norm(x, (D,), W[f"h{l}.ln2.w"], W[f"h{l}.ln2.b"], EPS)
            x = x + gelu_new(h @ W[f"h{l}.mlp.w1"] + W[f"h{l}.mlp.b1"]) @ W[f"h{l}.mlp.w2"] + W[f"h{l}.mlp.b2"]
        if every is not None:
            every(r, x)
    return readout(W, x)


def strict(logits, Y):
    """logits [N, T, VOCAB], Y [N, L]: strict per row (correct logit above all 12 others at every answer position; ties wrong) and margin per row."""
    Y = torch.as_tensor(Y)
    lg = logits[:, 0:2 * Y.shape[1]:2]
    c = lg.gather(-1, Y[..., None])[..., 0]
    o = lg.clone()
    o.scatter_(-1, Y[..., None], float("-inf"))
    m = (c - o.max(-1).values).min(1).values
    return m > 0, m


def nonzero(W):
    return int(sum(int((v != 0).sum()) for v in W.values()))


def dense(cfg):
    return int(sum(int(np.prod(s)) for s in shapes(cfg).values()))


def save(path, W, cfg):
    """byte-for-byte deterministic."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        def add(name, raw):
            i = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            z.writestr(i, raw)
        add("config.json", json.dumps({k: cfg[k] for k in CFG_KEYS}, sort_keys=True).encode())
        for k in sorted(W):
            b = io.BytesIO()
            np.save(b, np.asarray(W[k], dtype=np.float32), allow_pickle=False)
            add(k + ".npy", b.getvalue())


def load(path, dtype=torch.float32):
    with zipfile.ZipFile(path) as z:
        cfg = json.loads(z.read("config.json"))
        W = {n[:-4]: torch.from_numpy(np.load(io.BytesIO(z.read(n)), allow_pickle=False)).to(dtype) for n in z.namelist() if n.endswith(".npy")}
    return W, cfg
