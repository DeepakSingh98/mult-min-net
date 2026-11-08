"""Learned single-task Triplet-GMPNN baseline with the paper's recipe = clrs/examples/run.py main() (single algorithm,
not chunked): online training sampler cycling sizes [4, 7, 11, 13, 16] (string matchers: n=20 with needle 1..8),
ER p in {0.1..0.9}, random pos, Adam 1e-3, clip 1.0, batch 32, 10,000 steps; validate every 50 steps on n=16 (20),
keep the best-validation parameters, test them. Differences from run.py: the test set is regenerated with the
official sampler and seed (CLRS30 test seed 3, n=64) instead of downloading the CLRS-30 files, matmuls are full fp32,
and `init` (optional) starts from given parameters instead of the Haiku initialisers (SGD-from-programmed follow-up).
"""
import json
import os
import time

import numpy as np
import jax

import common as C
import clrs


def _sampler(algo, length, rng, split, batch_size, multiplier, randomize_pos, sampler_kwargs):
    num = clrs.CLRS30[split]["num_samples"] * multiplier
    sampler, spec = clrs.build_sampler(algo, seed=rng.randint(2 ** 32), num_samples=num, length=length,
                                       **sampler_kwargs)

    def it():
        while True:
            yield sampler.next(batch_size)
    spec, s = C._postprocess(algo, spec, it(), randomize_pos, rng)
    return s, num, spec


def _multi(sizes, rng, **kw):
    ss, tot, spec = [], 0, None
    for n in sizes:
        s, num, spec = _sampler(rng=rng, length=n, **kw)
        ss.append(s)
        tot += num

    def cyc():
        while True:
            for s in ss:
                yield next(s)
    return cyc(), tot, spec


def collect(model, sampler, count, key):
    outs, preds, done = [], [], 0
    while done < count:
        fb = next(sampler)
        key, sub = jax.random.split(key)
        p, _ = model.predict(sub, fb.features)
        outs.append(fb.outputs)
        preds.append(jax.device_get(p))
        done += fb.outputs[0].data.shape[0]
    cat = lambda *x: np.concatenate(x, 0)  # noqa: E731
    o = jax.tree_util.tree_map(cat, *outs)
    p = jax.tree_util.tree_map(cat, *preds)
    return {k: float(v) for k, v in clrs.evaluate(o, p).items()}


def run(cfg, out_dir, log=print):
    algo = cfg["algo"]
    seed = cfg.get("seed", 42)
    steps = cfg.get("steps", 10000)
    eval_every = cfg.get("eval_every", 50)
    rng = np.random.RandomState(seed)
    key = jax.random.PRNGKey(rng.randint(2 ** 32))
    train_lengths = [4, 7, 11, 13, 16]
    if algo in ("naive_string_matcher", "kmp_matcher"):
        train_lengths = [(max(train_lengths) * 5) // 4]
    kw = dict(sampler_kwargs=dict(p=tuple(0.1 + 0.1 * i for i in range(9)), length_needle=-8), algo=algo)
    # build_sampler drops kwargs a sampler does not accept (with a warning), as in run.py
    train_s, _, _ = _multi(train_lengths, rng, split="train", batch_size=32, multiplier=-1, randomize_pos=True, **kw)
    mult = clrs.CLRS_30_ALGS_SETTINGS[algo]["num_samples_multiplier"]
    val_s, val_n, spec = _multi([max(train_lengths)], rng, split="val", batch_size=32, multiplier=2 * mult,
                                randomize_pos=True, **kw)
    model = clrs.models.BaselineModel(spec=[spec], dummy_trajectory=[next(val_s)],
                                      processor_factory=C.processor_factory(), **C.MODEL_KW)
    best, best_params, hist = -1.0, None, []
    t0 = time.time()
    for step in range(steps):
        fb = next(train_s)
        if step == 0:
            model.init([fb.features], seed + 1)
            if cfg.get("init"):
                model.params = C.unflatten(dict(np.load(cfg["init"])))
                model.opt_state = model.opt.init(model.params)
                log(f"initialised from {cfg['init']}")
        key, sub = jax.random.split(key)
        loss = float(model.feedback(sub, fb))
        if step % eval_every == 0 or step == steps - 1:
            key, sub = jax.random.split(key)
            v = collect(model, val_s, val_n, sub)
            rec = {"step": step, "loss": loss, "val": v["score"], "t": round(time.time() - t0, 1)}
            if v["score"] > best or step == 0:
                best, best_params = v["score"], jax.device_get(model.params)
                rec["best"] = True
            hist.append(rec)
            if step % (eval_every * 10) == 0 or step == steps - 1:
                log(json.dumps(rec))
    flat = C.flatten(best_params)
    np.savez(os.path.join(out_dir, "best.npz"), **flat)
    np.savez(os.path.join(out_dir, "last.npz"), **C.flatten(jax.device_get(model.params)))
    json.dump(hist, open(os.path.join(out_dir, "val_history.json"), "w"))
    return {"algo": algo, "seed": seed, "steps": steps, "best_val": best, "train_s": round(time.time() - t0, 1)}
