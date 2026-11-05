"""Check a network on random problems (or all of them) under the strict criterion, with float32 weights and with
bf16-rounded weights (float32 computation).

    python verify.py weights/looped.npz                       100,000 problems, seed drawn from os.urandom
    python verify.py weights/*.npz --n 100000 --seed 12345    your own seed (numpy default_rng(seed); a, b in 1000..9999)
    python verify.py weights/two_layer.npz --exhaustive       all 81,000,000 operand pairs (hours on a laptop CPU)
    python verify.py weights/two_layer.npz --exhaustive --a-from 1000 --a-to 1100    one slice of a (a-to exclusive)

A problem is strictly right when, at all 8 answer positions, the correct token's logit is strictly above all 13 other
logits (a tie counts as wrong). "argmax wrong" uses torch's argmax instead, which breaks ties toward the lowest token id.
The min margin is the smallest (correct logit - best other logit) over all problems and digits checked.
"""
import argparse
import os
import time

import numpy as np
import torch

import arch


def check(W, cfg, batches):
    r = {"n": 0, "argmax_wrong": 0, "strict_wrong": 0, "min_margin": float("inf"), "examples": []}
    for a, b in batches:
        ok, m = arch.margins(W, a, b, cfg)
        bad = m <= 0
        r["n"] += len(a)
        r["argmax_wrong"] += int((~ok).sum())
        r["strict_wrong"] += int(bad.sum())
        r["min_margin"] = min(r["min_margin"], float(m.min()))
        for i in bad.nonzero().flatten()[:5].tolist():
            if len(r["examples"]) < 5:
                r["examples"].append((int(a[i]), int(b[i]), float(m[i])))
    return r


def random_batches(n, seed, size):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(1000, 10000, n), rng.integers(1000, 10000, n)
    for i in range(0, n, size):
        yield a[i:i + size], b[i:i + size]


def all_batches(a_from, a_to):
    ys = np.arange(1000, 10000)
    for x in range(a_from, a_to):
        yield np.full(9000, x), ys


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("weights", nargs="+")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch", type=int, default=10_000)
    p.add_argument("--exhaustive", action="store_true")
    p.add_argument("--a-from", type=int, default=1000)
    p.add_argument("--a-to", type=int, default=10000)
    args = p.parse_args()
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    torch.set_grad_enabled(False)
    bad = 0
    for path in args.weights:
        cfg, W = arch.load(path)
        what = (f"all pairs with a in [{args.a_from}, {args.a_to})" if args.exhaustive
                else f"{args.n:,} random problems, seed {seed}")
        print(f"{path}: {arch.nonzero(W):,} nonzero weights, dense size {arch.dense(cfg):,}, config {cfg}")
        print(f"  {what}")
        for name, Wx in (("fp32", W), ("bf16-rounded", arch.to_bf16(W))):
            t0 = time.time()
            batches = (all_batches(args.a_from, args.a_to) if args.exhaustive
                       else random_batches(args.n, seed, args.batch))
            r = check(Wx, cfg, batches)
            bad += r["strict_wrong"]
            print(f"  {name:13s} n {r['n']:,}  strict wrong {r['strict_wrong']:,}  argmax wrong {r['argmax_wrong']:,}"
                  f"  min margin {r['min_margin']:.6g}  ({time.time() - t0:.1f} s)")
            for a, b, m in r["examples"]:
                print(f"    e.g. {a} x {b}: margin {m:.6g}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
