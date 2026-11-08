"""TEACH-WRITTEN: an ordinary Triplet-GMPNN from random init, trained by Adam on the official recipe, with optional matching of a
written teacher's per-step latents. Recipe as clrs/examples/run.py single task (and cwp/train.py): online sampler sizes
4, 7, 11, 13, 16, random pos, Adam 1e-3, clip 1.0, batch 32, 10,000 steps, validate every 50 on n=16.

Arms (cfg["arm"]; wiring in ARMS):
  nohint     outputs only; no hint encoders or decoders
  hints      official hints (encoded, decoded, soft, no teacher forcing): the baseline to beat
  lat        no hint loss; per-step latent matching. Hint encoders and decoders stay, as in the teacher, because the teacher keeps
             array order in its fed-back hints; only the latent loss and the output loss train them
  lathints   official hints plus latent matching
  latanneal  lat, with the latent weight falling linearly to zero at cfg["anneal_end"] (default mid-training)
All arms carry detached hint probes (TNet.probe): decoders of the official form fit to the frozen latents. They have their own
optimizer and no clip shared with the model, so they cannot change training. They give one hint accuracy for every arm.

Checkpoints: `best` = latest step at the best validation score, `first` = earliest (the official rule), `last`.
"""
import argparse
import json
import os
import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import common as C
import train as T
import clrs
from clrs._src import decoders, losses
from tnet import TNet

GROUPS = ("hidden", "tri", "node", "edge", "graph")
ARMS = {"nohint": dict(enc=False, dec=False, hint=False, lat=False), "hints": dict(enc=True, dec=True, hint=True, lat=False),
        "lat": dict(enc=True, dec=True, hint=False, lat=True), "lathints": dict(enc=True, dec=True, hint=True, lat=True),
        "latanneal": dict(enc=True, dec=True, hint=False, lat=True), "lat_noloop": dict(enc=False, dec=False, hint=False, lat=True)}
H = C.MODEL_KW["hidden_dim"]


def nb_dims_of(fb):
    return [{d.name: d.data.shape[-1] for d in list(fb.features.inputs) + list(fb.features.hints) + list(fb.outputs)}]


def net(spec, dims, enc, dec, **kw):
    return TNet([spec], H, enc, dec, C.processor_factory(), False, C.MODEL_KW["encoder_init"], 0.0, 0.0, "soft", dims, 1, False,
                name="net", **kw)


def slog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def to_target(g, x, norm):
    """x: teacher latent, last axis units. Frozen normalization, chosen on dev."""
    u = norm[g]
    y = jnp.take(x, u["units"], axis=-1)
    if norm["mode"] == "rms":
        return y / u["rms"]
    if norm["mode"] == "slog":
        return slog(y)
    if norm["mode"] == "slogz":
        return slog(y) / u["slog_rms"]
    raise ValueError(norm["mode"])


def load_norm(path, mode, groups):
    d = np.load(path)
    n = {"mode": mode}
    for g in GROUPS:
        n[g] = dict(units=jnp.asarray(d[g + "_units"]), rms=jnp.asarray(d[g + "_rms"], jnp.float32),
                    slog_rms=jnp.asarray(d[g + "_slog_rms"], jnp.float32))
    n["groups"] = [g for g in GROUPS if groups.get(g, 0) > 0 and int(d[g + "_units"].size) > 0]
    return n


def teacher_fn(spec, dims, norm):
    def f(features):
        def on_step(i, lat):
            out = {g: to_target(g, lat[g], norm) for g in norm["groups"]}
            return out
        o, h, ex = net(spec, dims, True, True, on_step=on_step)([features], True, 0, True, False)
        return ex["step"], h, o
    return hk.transform(f)


def student_fn(spec, dims, a, norm, groups, cfg):
    def f(features, repred, tstate, tf_p):
        lengths = features.lengths
        force, inject = None, None
        if tstate is not None and cfg.get("state_tf", 0) > 0:
            def force(i):
                """one coin per instance per step: this step starts from the teacher's full state, or from the student's own"""
                return jax.random.bernoulli(hk.next_rng_key(), tf_p, (features.lengths.shape[0],)) & (i > 0)

            def inject(i, hidden, coin):
                prev = tstate["hidden"][jnp.maximum(i - 1, 0)]
                full = hidden.at[..., norm["hidden"]["units"]].set(prev)
                return jnp.where(coin[:, None, None], full, hidden)

        def on_step(i, lat):
            if tstate is None:
                return {}
            live = (lengths > i + 1).astype(jnp.float32)
            out = {}
            for g in norm["groups"]:
                s = jnp.take(lat[g], norm[g]["units"], axis=-1)
                t = tstate[g][i]
                e = jnp.mean((s - t) ** 2, axis=tuple(range(1, s.ndim)))
                out[g] = jnp.stack([jnp.sum(e * live), jnp.sum(live)])
            return out
        return net(spec, dims, a["enc"], a["dec"], on_step=on_step if norm is not None else None, inject=inject, force=force, probe=True)(
            [features], repred, 0, True, False)
    return hk.transform(f)


def split(params):
    pr = {k: v for k, v in params.items() if "probe_" in k}
    st = {k: v for k, v in params.items() if "probe_" not in k}
    return st, pr


def probe_loss(fb, probe_preds, nb_nodes):
    steps = [dict(zip(probe_preds, x)) for x in zip(*probe_preds.values())]
    tot = 0.0
    for truth in fb.features.hints:
        tot += losses.hint_loss(truth=truth, preds=[x[truth.name] for x in steps], lengths=fb.features.lengths, nb_nodes=nb_nodes)
    return tot


def as_steps(d):
    return [dict(zip(d, x)) for x in zip(*d.values())]


def evaluate(apply_eval, params, algo, spec, n, num, bs, hints=True, dev=False):
    """test set: official sampler, test seed 3, pos = i/n. dev=True: seed 1001, random pos, never the test set."""
    t0 = time.time()
    fb, _ = C.dataset(algo, n, num, seed=1001, random_pos=True, p=(0.5,)) if dev else C.test_set(algo, n, num)
    outs, own, probe = [], [], []
    for lo in range(0, num, bs):
        b = C.trim_time(C.slice_fb(fb, lo, min(num, lo + bs)))
        o, h, ex = apply_eval(params, b.features)
        outs.append(jax.device_get(decoders.postprocess(spec, o, sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True)))
        own.append(jax.device_get(h) if h else None)
        probe.append(as_steps(jax.device_get(ex["probe"])))
    preds = {k: C._cat([x[k] for x in outs]) for k in outs[0]}
    sc = clrs.evaluate(fb.outputs, preds)
    ex = C.per_instance_exact(fb.outputs, preds)
    r = dict(n=n, num=num, score=float(sc["score"]), exact=float(ex.mean()), max_len=int(np.max(fb.features.lengths)))
    if hints:
        tr = C.hint_trajectory(fb, spec, probe, bs)
        r.update(probe_step_acc=tr["per_hint"], probe_traj=tr["traj_exact"], probe_strict=float(np.mean(tr["inst_ok"] & ex)))
        if own[0] is not None:
            tr = C.hint_trajectory(fb, spec, own, bs)
            r.update(own_step_acc=tr["per_hint"], own_traj=tr["traj_exact"], own_strict=float(np.mean(tr["inst_ok"] & ex)))
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def run(cfg, out_dir, log):
    algo, armname, seed = cfg["algo"], cfg["arm"], cfg["seed"]
    a = ARMS[armname]
    steps, eval_every = cfg.get("steps", 10000), cfg.get("eval_every", 50)
    resume = os.path.join(out_dir, "resume.pkl")     # Spot preemption: full state; the sampler stream restarts from seed and step
    prior = None
    if os.path.exists(resume):
        prior = pickle.load(open(resume, "rb"))
        assert prior["cfg"] == cfg, "resume state is from another config"
    rng = np.random.RandomState(seed if prior is None else seed * 1000003 + prior["step"])
    key = jax.random.PRNGKey(rng.randint(2 ** 32))
    train_lengths = [4, 7, 11, 13, 16]
    kw = dict(sampler_kwargs=dict(p=tuple(0.1 + 0.1 * i for i in range(9)), length_needle=-8), algo=algo)
    train_s, _, _ = T._multi(train_lengths, rng, split="train", batch_size=32, multiplier=-1, randomize_pos=True, **kw)
    mult = clrs.CLRS_30_ALGS_SETTINGS[algo]["num_samples_multiplier"]
    val_s, val_n, spec = T._multi([16], rng, split="val", batch_size=32, multiplier=2 * mult, randomize_pos=True, **kw)
    probe_fb = next(val_s)
    dims = nb_dims_of(probe_fb)
    groups = cfg.get("groups", dict(hidden=1.0, tri=1.0, node=1.0, edge=1.0, graph=1.0))
    norm = load_norm(cfg["norm"], cfg.get("norm_mode", "slogz"), groups) if a["lat"] else None
    teach_sizes = set(cfg.get("teach_sizes", train_lengths))
    sf = student_fn(spec, dims, a, norm, groups, cfg)
    ev = student_fn(spec, dims, a, None, groups, cfg)
    apply_eval = jax.jit(lambda p, f: ev.apply(p, jax.random.PRNGKey(0), f, True, None, 0.0))
    params = sf.init(jax.random.PRNGKey(seed + 1), probe_fb.features, False, None, 0.0)
    st, pr = split(params)
    lr = cfg.get("lr", 1e-3)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-lr))
    popt = optax.adam(1e-3)
    ost, opr = opt.init(st), popt.init(pr)
    tparams, tf = None, None
    if a["lat"]:
        P = {k: np.asarray(v) for k, v in np.load(cfg["program"]).items()}
        tf = teacher_fn(spec, dims, norm)
        ref = C.flatten(tf.init(jax.random.PRNGKey(0), probe_fb.features))
        tparams = C.unflatten({k: P.get(k, np.zeros_like(ref[k])) for k in ref})
        check = jax.jit(lambda f: tf.apply(tparams, None, f))

    def loss_fn(st, pr, fb, tstate, lam, tf_p, rk):
        p = {**st, **pr}
        o, h, ex = sf.apply(p, rk, fb.features, False, tstate, tf_p)
        nb = fb.features.inputs[0].data.shape[1]
        off = 0.0
        for truth in fb.outputs:
            off += losses.output_loss(truth=truth, pred=o[truth.name], nb_nodes=nb)
        if a["hint"]:
            for truth in fb.features.hints:
                off += losses.hint_loss(truth=truth, preds=[x[truth.name] for x in h], lengths=fb.features.lengths, nb_nodes=nb)
        lat, per = 0.0, {}
        if tstate is not None:
            for g in norm["groups"]:
                per[g] = jnp.sum(ex["step"][g][:, 0]) / jnp.maximum(jnp.sum(ex["step"][g][:, 1]), 1.0)     # pooled, as hint_loss
                lat += groups[g] * per[g]
        pl = probe_loss(fb, ex["probe"], nb)
        return off + lam * lat + pl, (off, per, pl)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    @jax.jit
    def step_no(st, pr, ost, opr, fb, rk):
        (_, aux), (gs, gp) = grad_fn(st, pr, fb, None, 0.0, 0.0, rk)
        u, ost = opt.update(gs, ost, st)
        v, opr = popt.update(gp, opr, pr)
        return optax.apply_updates(st, u), optax.apply_updates(pr, v), ost, opr, aux

    @jax.jit
    def step_lat(st, pr, ost, opr, fb, rk, lam, tf_p):
        tstate, _, _ = tf.apply(tparams, None, fb.features)
        tstate = jax.lax.stop_gradient(tstate)
        (_, aux), (gs, gp) = grad_fn(st, pr, fb, tstate, lam, tf_p, rk)
        u, ost = opt.update(gs, ost, st)
        v, opr = popt.update(gp, opr, pr)
        return optax.apply_updates(st, u), optax.apply_updates(pr, v), ost, opr, aux

    def val(p, k):
        outs, preds, done = [], [], 0
        while done < val_n:
            fb = next(val_s)
            o, _, _ = apply_eval(p, fb.features)
            outs.append(fb.outputs)
            preds.append(jax.device_get(decoders.postprocess(spec, o, sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True)))
            done += fb.outputs[0].data.shape[0]
        cat = lambda *x: np.concatenate(x, 0)  # noqa: E731
        return float(clrs.evaluate(jax.tree_util.tree_map(cat, *outs), jax.tree_util.tree_map(cat, *preds))["score"])

    lam0 = cfg.get("lam", 1.0)
    end = cfg.get("anneal_end", steps // 2) if armname.startswith("latanneal") else None
    tf0 = cfg.get("state_tf", 0.0)
    best, bp, fp, hist, nan_steps = -1.0, None, None, [], 0
    best_step, first_step, teacher_checks = None, None, []
    info = dict(cfg=cfg, device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind, wiring=a)
    t0, start, resumes = time.time(), 0, []
    if prior is not None:
        st, pr, ost, opr, key = prior["st"], prior["pr"], prior["ost"], prior["opr"], prior["key"]
        best, bp, fp, hist, nan_steps = prior["best"], prior["bp"], prior["fp"], prior["hist"], prior["nan_steps"]
        best_step, first_step, teacher_checks = prior["best_step"], prior["first_step"], prior["teacher_checks"]
        start, resumes = prior["step"], prior["resumes"] + [prior["step"]]
        t0 -= prior["train_s"]
        log(json.dumps(dict(resumed=start)))
    every = cfg.get("resume_every", 250)
    for step in range(start, steps):
        fb = next(train_s)
        key, sub = jax.random.split(key)
        n = fb.features.inputs[0].data.shape[1]
        lam = lam0 if end is None else lam0 * max(0.0, 1.0 - step / end)
        if a["lat"] and n in teach_sizes and lam > 0:
            tf_p = tf0 * max(0.0, 1.0 - step / (cfg.get("tf_end") or steps))
            st, pr, ost, opr, aux = step_lat(st, pr, ost, opr, fb, sub, jnp.float32(lam), jnp.float32(tf_p))
        else:
            st, pr, ost, opr, aux = step_no(st, pr, ost, opr, fb, sub)
        off, per, pl = aux
        if a["lat"] and step % 997 == 0:     # the teacher really is exact on what it teaches (sizes cycle through this modulus)
            _, h, o = check(fb.features)
            o = decoders.postprocess(spec, o, sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True)
            ok = bool(C.per_instance_exact(fb.outputs, jax.device_get(o)).all()) and C.hint_trajectory(fb, spec, [jax.device_get(h)], 32)["traj_exact"] == 1.0
            teacher_checks.append([step, n, ok])
        if not np.isfinite(float(off)):
            nan_steps += 1
        if step % eval_every == 0 or step == steps - 1:
            key, sub = jax.random.split(key)
            p = {**st, **pr}
            v = val(p, sub)
            v = v if np.isfinite(v) else 0.0
            rec = dict(step=step, off=float(off), probe=float(pl), val=v, t=round(time.time() - t0, 1), nan=nan_steps,
                       **{"lat_" + k: float(x) for k, x in per.items()})
            if v > best or step == 0:
                fp, first_step = jax.device_get(p), step
            if v >= best or step == 0:
                bp, best_step = jax.device_get(p), step
                rec["best"] = True
            best = max(best, v)
            hist.append(rec)
            if step % (eval_every * 10) == 0 or step == steps - 1:
                log(json.dumps(rec))
        if (step + 1) % every == 0 or step == steps - 1:
            state = dict(cfg=cfg, step=step + 1, key=key, best=best, bp=bp, fp=fp, hist=hist, nan_steps=nan_steps, best_step=best_step,
                         first_step=first_step, teacher_checks=teacher_checks, resumes=resumes, train_s=time.time() - t0,
                         **jax.device_get(dict(st=st, pr=pr, ost=ost, opr=opr)))
            with open(resume + ".tmp", "wb") as f:
                pickle.dump(state, f)
            os.replace(resume + ".tmp", resume)
    last = jax.device_get({**st, **pr})
    json.dump(hist, open(os.path.join(out_dir, "val_history.json"), "w"))
    for nm, x in (("best", bp), ("first", fp), ("last", last)):
        if cfg.get("save", True):
            np.savez(os.path.join(out_dir, nm + ".npz"), **C.flatten(x))
    res = dict(info, best_val=best, best_step=best_step, first_step=first_step, train_s=round(time.time() - t0, 1), nan_steps=nan_steps,
               final=hist[-1], teacher_checks=teacher_checks, resumes=resumes, tests=[])
    path = os.path.join(out_dir, "result.json")
    json.dump(res, open(path, "w"), indent=1)
    for row in cfg.get("tests", [["best", 16, 32, 32], ["best", 64, 32, 32], ["best", 128, 8, 8],
                                 ["first", 16, 32, 32], ["first", 64, 32, 32], ["first", 128, 8, 8]]):
        ck, n, num, bs = row[:4]
        dev = len(row) > 4 and row[4] == "dev"
        try:
            r = evaluate(apply_eval, dict(best=bp, first=fp, last=last)[ck], algo, spec, n, num, bs, dev=dev)
        except Exception as e:  # noqa: BLE001
            r = dict(n=n, num=num, error=repr(e)[:400])
        r["ckpt"] = ck
        r["set"] = "dev" if dev else "test"
        res["tests"].append(r)
        json.dump(res, open(path, "w"), indent=1)
        log(json.dumps({k: r.get(k) for k in ("ckpt", "n", "score", "exact", "probe_traj", "own_traj", "error", "wall_s")}))
    os.remove(resume)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(m):
        logf.write(m + "\n")
        logf.flush()
    run(json.loads(args.cfg), args.out, log)
    print("done", args.out, flush=True)
