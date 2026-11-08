"""MULT-LOGDEPTH Part 1: from-random training of the any-length architecture and of a Neural GPU reimplementation.

Task: n-digit x m-digit multiplication in base `base`. Budget: `tokens` training tokens (sequence length x batch, summed over
updates), the same for both models. Curriculum: `cur` grows from 1 to `lmax` when strict exact at (cur, cur) reaches 0.95.
kind "tf": the equations of multgap/arch_anylen.py (causal pre-LN blocks, gelu_new, ALiBi bias, tied output, K passes of the same
n_layer blocks), format BOS a * b EOS c, teacher-forced answer digits; task "add" is the positive control of this pipeline.
kind "ngpu": the encoder path of the authors' code (tensorflow/models v1.13.0 research/neural_gpu, read for this port), plus the 2015 paper's relaxation pull;
base curriculum 2 -> 4 -> 10 for decimal (Price et al. 2016). Reimplementation; LR decay on plateaus and the later decoder path are not ported.
Advance needs two consecutive evals on 256 problems at the arm's threshold; after lmax is learned a further `consolidate` tokens are trained.
Strict = every answer digit is the argmax. Output: result JSON with curve, test table and every resource count; state saved for preemption.
"""
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MUL, EOS, BOS = 10, 11, 12
VOCAB = 14
PLUS = 13


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def digits(v, n, base):
    return [(v // base ** i) % base for i in range(n)]


def sample_pairs(N, n, m, base, rng):
    """arrays of digits, LSB first, [N, n] and [N, m], uniform digits (leading zeros allowed)."""
    return rng.integers(0, base, (N, n)), rng.integers(0, base, (N, m))


def product(A, B, base, task="mul"):
    """digit arrays -> [N, n+m] LSB-first digits of A*B (or A+B, width max(n,m)+1); python big ints, any length."""
    N, n, m = len(A), A.shape[1], B.shape[1]
    w = n + m if task == "mul" else max(n, m) + 1
    out = np.zeros((N, w), dtype=np.int64)
    for i in range(N):
        a = sum(int(d) * base ** j for j, d in enumerate(A[i]))
        b = sum(int(d) * base ** j for j, d in enumerate(B[i]))
        c = a * b if task == "mul" else a + b
        for j in range(w):
            c, out[i, j] = divmod(c, base)
    return out


class Looped(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        D, H, DH, M = c["d_model"], c["n_head"], c["d_head"], c["d_mlp"]
        s.wte = nn.Parameter(torch.randn(VOCAB, D) * 0.02)
        s.ln = nn.ModuleList()
        s.qkv = nn.ModuleList()
        s.wo = nn.ModuleList()
        s.m1 = nn.ModuleList()
        s.m2 = nn.ModuleList()
        res = 0.02 / math.sqrt(2 * c["n_layer"])
        for _ in range(c["n_layer"]):
            s.ln.append(nn.ModuleList([nn.LayerNorm(D, 1e-5), nn.LayerNorm(D, 1e-5)]))
            s.qkv.append(nn.Linear(D, 3 * H * DH))
            s.wo.append(nn.Linear(H * DH, D))
            s.m1.append(nn.Linear(D, M))
            s.m2.append(nn.Linear(M, D))
        for mod in list(s.qkv) + list(s.m1):
            nn.init.normal_(mod.weight, std=0.02)
            nn.init.zeros_(mod.bias)
        for mod in list(s.wo) + list(s.m2):
            nn.init.normal_(mod.weight, std=res)
            nn.init.zeros_(mod.bias)
        s.lnf = nn.LayerNorm(D, 1e-5)
        s.register_buffer("slopes", torch.tensor(c["alibi"], dtype=torch.float32))   # [n_layer, n_head]

    def forward(s, tok, K):
        c = s.c
        N, T = tok.shape
        D, H, DH = c["d_model"], c["n_head"], c["d_head"]
        A = H * DH
        x = s.wte[tok]
        pos = torch.arange(T, device=tok.device)
        dist = (pos[:, None] - pos[None, :]).float()
        bias = [(-s.slopes[l][:, None, None] * dist).masked_fill(dist < 0, float("-inf")) for l in range(c["n_layer"])]
        for r in range(K):
            for l in range(c["n_layer"]):
                h = s.ln[l][0](x)
                q, k, v = (t.view(N, T, H, DH).transpose(1, 2) for t in s.qkv[l](h).split(A, dim=-1))
                a = ((q @ k.transpose(-1, -2)) / math.sqrt(DH) + bias[l]).softmax(-1)
                x = x + s.wo[l]((a @ v).transpose(1, 2).reshape(N, T, A))
                x = x + s.m2[l](gelu_new(s.m1[l](s.ln[l][1](x))))
        return s.lnf(x) @ s.wte.T


def tf_K(rule, n, m):
    return dict(sys=2 * n + 3 * m + 2, fast=n + m + 2, two=2 * (n + m) + 2)[rule]


def tf_tokens(A, B, C, task, dev):
    """BOS a * b EOS c."""
    N = len(A)
    op = MUL if task == "mul" else PLUS
    t = np.concatenate([np.full((N, 1), BOS), A, np.full((N, 1), op), B, np.full((N, 1), EOS), C], 1)
    return torch.as_tensor(t, device=dev, dtype=torch.long), A.shape[1] + B.shape[1] + 2


def tf_loss(model, A, B, C, cfg, dev, task):
    tok, e = tf_tokens(A, B, C, task, dev)
    K = tf_K(cfg["K_rule"], A.shape[1], B.shape[1])
    lg = model(tok[:, :-1], K)[:, e:]
    return F.cross_entropy(lg.reshape(-1, VOCAB), tok[:, e + 1:].reshape(-1)), lg, tok[:, e + 1:]


@torch.no_grad()
def tf_score(model, n, m, cfg, N, base, rng, dev, task, greedy=False, bs=32):
    ok = hit = tot = 0
    gok = 0
    for i in range(0, N, bs):
        k = min(bs, N - i)
        A, B = sample_pairs(k, n, m, base, rng)
        C = product(A, B, base, task)
        _, lg, y = tf_loss(model, A, B, C, cfg, dev, task)
        p = lg.argmax(-1)
        ok += int((p == y).all(1).sum())
        hit += int((p == y).sum())
        tot += y.numel()
        if greedy:
            g = np.zeros_like(C)
            for j in range(C.shape[1]):
                _, l2, _ = tf_loss(model, A, B, g, cfg, dev, task)
                g[:, j] = l2[:, j].argmax(-1).cpu().numpy()
            gok += int((g == C).all(1).sum())
    r = dict(n=n, m=m, N=N, exact=ok / N, digit=hit / tot)
    if greedy:
        r["greedy_exact"] = gok / N
    return r


class CGRU(nn.Module):
    """conv_gru of neural_gpu.py: reset and gate are sigmoid cut at 1.2, candidate tanh; biases 1, 1, 0; orthogonal init gain 1.44."""
    def __init__(s, nmaps):
        super().__init__()
        s.g = nn.Conv2d(nmaps, nmaps, 3, padding=1)
        s.r = nn.Conv2d(nmaps, nmaps, 3, padding=1)
        s.c = nn.Conv2d(nmaps, nmaps, 3, padding=1)
        for mod, b in ((s.g, 1.0), (s.r, 1.0), (s.c, 0.0)):
            nn.init.orthogonal_(mod.weight, gain=1.44)
            nn.init.constant_(mod.bias, b)

    def forward(s, x):
        g = (1.2 * torch.sigmoid(s.g(x)) - 0.1).clamp(0, 1)
        r = (1.2 * torch.sigmoid(s.r(x)) - 0.1).clamp(0, 1)
        return g * x + (1 - g) * torch.tanh(s.c(r * x))


class NGPU(nn.Module):
    """encoder path of neural_gpu.py (beam_size 0, tensorflow/models v1.13.0): embedding, 1x1 conv with channel LayerNorm into row 0, tape-width steps of
    nconvs = 2 CGRUs, dropout 0.1 x 3 / sqrt(width), output read from row 1. Relaxation pull (rx > 1) is from the 2015 paper; that code tag has none."""
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.em = nn.Embedding(VOCAB, 64)
        nn.init.uniform_(s.em.weight, -1.7, 1.7)
        s.inp = nn.Conv2d(64, c["nmaps"], 1)
        s.nrm = nn.LayerNorm(c["nmaps"], 1e-5)
        s.sets = nn.ModuleList([nn.ModuleList([CGRU(c["nmaps"]), CGRU(c["nmaps"])]) for _ in range(c["rx"])])
        s.out = nn.Conv2d(c["nmaps"], VOCAB, 1)

    def forward(s, tok):
        B, w = tok.shape
        e = s.em(tok).permute(0, 2, 1)[:, :, None]                     # [B, 64, 1, w]
        e = s.nrm(s.inp(e).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.cat([e, torch.zeros(B, e.shape[1], s.c["height"] - 1, w, device=e.device)], 2)
        drop = 0.1 * 3.0 / math.sqrt(w)
        for t in range(w):
            g = s.sets[t % s.c["rx"]]
            x = g[1](g[0](F.dropout(x, min(drop, 0.5), s.training)))
        return s.out(x)[:, :, 1].permute(0, 2, 1)                      # [B, w, VOCAB]

    def pull(s):
        v = 0.0
        ps = [list(g.parameters()) for g in s.sets]
        for i in range(len(ps) - 1):
            for a, b in zip(ps[i], ps[i + 1]):
                v = v + ((a - b) ** 2).sum()
        return v


def ngpu_batch(A, B, C, dev, task):
    """tape a * b, width n+m+1, output LSB first from column 0 (all n+m digits, no padding needed)."""
    N, n, m = len(A), A.shape[1], B.shape[1]
    w = n + m + 1
    op = MUL if task == "mul" else PLUS
    t = np.zeros((N, w), dtype=np.int64)
    t[:, :n] = A
    t[:, n] = op
    t[:, n + 1:n + 1 + m] = B
    y = np.zeros((N, w), dtype=np.int64)
    y[:, :C.shape[1]] = C
    return torch.as_tensor(t, device=dev), torch.as_tensor(y, device=dev), C.shape[1]


@torch.no_grad()
def ngpu_score(model, n, m, N, base, rng, dev, task, bs=32):
    ok = hit = tot = 0
    for i in range(0, N, bs):
        k = min(bs, N - i)
        A, B = sample_pairs(k, n, m, base, rng)
        C = product(A, B, base, task)
        t, y, w = ngpu_batch(A, B, C, dev, task)
        p = model(t).argmax(-1)
        ok += int((p[:, :w] == y[:, :w]).all(1).sum())
        hit += int((p[:, :w] == y[:, :w]).sum())
        tot += y[:, :w].numel()
    return dict(n=n, m=m, N=N, exact=ok / N, digit=hit / tot)


def save_state(path, uri, st):
    torch.save(st, path + ".tmp")
    os.replace(path + ".tmp", path)
    if uri:
        subprocess.run(["gcloud", "storage", "cp", path, uri], capture_output=True, timeout=300)


def load_state(path, uri):
    if uri and not os.path.exists(path):
        subprocess.run(["gcloud", "storage", "cp", uri, path], capture_output=True, timeout=300)
    return torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None


def train(job, dev="cuda", state_uri=None, tick=None):
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"] + 1000)
    tf = job["kind"] == "tf"
    task, lmax = job.get("task", "mul"), job["lmax"]
    bases = job.get("bases", [job["base"]])
    model = (Looped(job) if tf else NGPU(job)).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=job["lr"], eps=1e-8 if tf else 1e-3)
    clip = 1.0 if tf else 4.0
    adv = job.get("advance", 0.95 if tf else 0.7)
    path = f"/tmp/{job['id']}.state"
    st = load_state(path, state_uri)
    step, tokens, cur, bi, curve, pull, t_prev = 0, 0, 1, 0, [], job.get("pull", 0.0), 0.0
    best, streak, end_at, blocks = None, 0, None, 0
    cur = job.get("cur0", 1)
    if st:
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step, tokens, cur, bi, curve, pull, t_prev, best = st["step"], st["tokens"], st["cur"], st["bi"], st["curve"], st["pull"], st["seconds"], st["best"]
        streak, end_at, blocks = st["streak"], st["end_at"], st["blocks"]
        rng.bit_generator.state = st["rng"]
    t0 = time.time()
    last_save = time.time()
    def score(n, m, N, r, base, **kw):
        return (tf_score(model, n, m, job, N, base, r, dev, task, **kw) if tf else ngpu_score(model, n, m, N, base, r, dev, task))
    stop = "tokens"
    while True:
        if tokens >= job["tokens"] or (end_at is not None and tokens >= end_at):
            stop = "tokens" if end_at is None else "consolidated"
            break
        if step >= job["max_steps"]:
            stop = "max_steps"
            break
        if time.time() - t0 + t_prev >= job["max_seconds"]:
            stop = "max_seconds"
            break
        base = bases[bi]
        if rng.random() < 0.8:
            n, m = int(rng.integers(1, cur + 1)), int(rng.integers(1, cur + 1))
            if rng.random() < 0.5:
                n = cur
        else:
            n, m = int(rng.integers(1, lmax + 1)), int(rng.integers(1, lmax + 1))
        A, B = sample_pairs(job["batch"], n, m, base, rng)
        C = product(A, B, base, task)
        for g in opt.param_groups:
            g["lr"] = job["lr"] * min(1.0, (step + 1) / 500)
        if tf:
            loss, _, _ = tf_loss(model, A, B, C, job, dev, task)
            tokens += job["batch"] * (A.shape[1] + B.shape[1] + 2 + C.shape[1])
            blocks += tf_K(job["K_rule"], n, m) * job["n_layer"]
        else:
            t, y, _ = ngpu_batch(A, B, C, dev, task)
            loss = F.cross_entropy(model(t).reshape(-1, VOCAB), y.reshape(-1), label_smoothing=0.1)
            blocks += t.shape[1] * 2
            if job["rx"] > 1 and pull > 0:
                loss = loss + pull * model.pull()
            tokens += job["batch"] * t.shape[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g2 = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(g2):
            opt.zero_grad(set_to_none=True)
        else:
            if job.get("grad_noise", 0.0) > 0:
                for p in model.parameters():
                    p.grad.add_(torch.randn_like(p.grad) * job["grad_noise"] / (1 + step) ** 0.275)
            opt.step()
        step += 1
        if step % job["eval_every"] == 0:
            model.eval()
            r = np.random.default_rng(job["seed"] + 77 + step)
            e = score(cur, cur, 256, r, base)
            ev8 = score(lmax, lmax, 64, r, bases[-1])
            curve.append(dict(step=step, tokens=tokens, cur=cur, base=base, loss=float(loss), exact=e["exact"], digit=e["digit"],
                              lmax_exact=ev8["exact"], lmax_digit=ev8["digit"], pull=pull, seconds=round(time.time() - t0 + t_prev, 1)))
            model.train()
            rank = (bi, cur, e["exact"], e["digit"])
            if best is None or rank >= tuple(best["rank"]):
                best = dict(rank=list(rank), step=step, cur=cur, base=base, exact=e["exact"], digit=e["digit"], lmax_exact=ev8["exact"], lmax_digit=ev8["digit"])
            streak = streak + 1 if e["exact"] >= adv else 0
            if streak >= 2:
                streak = 0
                if cur < lmax:
                    cur += 1
                elif bi + 1 < len(bases):
                    bi, cur = bi + 1, max(1, cur // 2)
                pull = pull * job.get("pull_incr", 1.0)
            if end_at is None and bi == len(bases) - 1 and cur == lmax and e["exact"] >= 0.99:
                end_at = tokens + job.get("consolidate", 40_000_000)
            if tick:
                tick(step, tokens, cur, bi, e)
        if time.time() - last_save > 900:
            save_state(path, state_uri, dict(model=model.state_dict(), opt=opt.state_dict(), step=step, tokens=tokens, cur=cur, bi=bi, curve=curve,
                                             pull=pull, seconds=time.time() - t0 + t_prev, best=best, rng=rng.bit_generator.state, streak=streak, end_at=end_at, blocks=blocks))
            last_save = time.time()
    model.eval()
    seconds = time.time() - t0 + t_prev
    tr = np.random.default_rng(job["test_seed"])
    test = []
    weak = False
    for (n, m, N, gr) in job["tests"]:
        t1 = time.time()
        if weak and max(n, m) > lmax:
            N = min(N, 32)
        with torch.no_grad():
            r = score(n, m, N, tr, bases[-1], **(dict(greedy=gr) if tf else {}))
        r["seconds"] = round(time.time() - t1, 1)
        test.append(r)
        if n == m == lmax:
            weak = r["exact"] < 0.5
    res = dict(job=job, params=params, steps=step, tokens=tokens, cur=cur, base_final=bases[bi], final_lmax=lmax, curve=curve, best=best, test=test, seconds=round(seconds, 1),
               stop=stop, blocks=blocks, reached=end_at is not None, device=str(dev))
    return res, model
