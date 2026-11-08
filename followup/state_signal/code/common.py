"""CLRS-WEIGHT-PROG shared code: the paper's single-task Triplet-GMPNN (official google-deepmind/clrs, pinned commit),
CLRS test sets regenerated with the official sampler and seed, and evaluation (micro-F1 = CLRS score, per-instance exact
match, per-step hint accuracy). Used by the editors' sandbox prelude, the final evaluator and the baseline trainer.

Model = clrs/examples/run.py defaults (Ibarz et al. 2022 recipe): hidden 128, triplet_gmpnn (gated MPNN + triplets,
8 triplet features), layer norm, encoded_decoded hints, soft hint re-prediction, xavier_on_scalars, no LSTM, no dropout.
Evaluation = BaselineModel.predict (repred=True: the model feeds back its own decoded hints; outputs hard-postprocessed).
All matmuls in full fp32 (jax_default_matmul_precision='highest').
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

import clrs  # noqa: E402
from clrs._src import decoders  # noqa: E402
from clrs._src import evaluation as _ev  # noqa: E402
from clrs._src import specs as _specs  # noqa: E402

CLRS_COMMIT = "6d5728b9efadf8648413c8c473929a733caacced"
ALGOS = ["quickselect", "kmp_matcher", "dfs", "bellman_ford"]
# Ibarz et al. 2022, Table 2: single-task OOD (n=64, fixed CLRS-30 test set) micro-F1, Triplet-GMPNN (ours), mean +- std
PAPER_TABLE2 = {"quickselect": (0.47, 0.25), "kmp_matcher": (19.51, 4.57), "dfs": (47.79, 4.19),
                "bellman_ford": (97.39, 0.19), "heapsort": (31.04, 5.82), "strongly_connected_components": (43.43, 3.15),
                "floyd_warshall": (48.52, 1.04)}
# clrs/examples/run.py
PRED_AS_INPUT_ALGOS = ["binary_search", "minimum", "find_maximum_subarray", "find_maximum_subarray_kadane",
                       "matrix_chain_order", "lcs_length", "optimal_bst", "activity_selector", "task_scheduling",
                       "naive_string_matcher", "kmp_matcher", "jarvis_march"]
MODEL_KW = dict(hidden_dim=int(__import__("os").environ.get("CLRS_HIDDEN", "128")), encode_hints=True, decode_hints=True, encoder_init="xavier_on_scalars", use_lstm=False,
                learning_rate=1e-3, grad_clip_max_norm=1.0, checkpoint_path="/tmp/cwp_ckpt", freeze_processor=False,
                dropout_prob=0.0, hint_teacher_forcing=0.0, hint_repred_mode="soft",
                nb_msg_passing_steps=int(__import__("os").environ.get("CLRS_NB_MP", "1")))
# CLRS-30-FULL: CLRS_NB_MP (default 1 = the paper setting) selects the library's own nb_msg_passing_steps knob; it is
# used only by the Kruskal 2-round architectural ablation (processor applied twice per hint step, shared weights).
TEST_SEED = 3          # samplers.CLRS30['test']['seed']: the official test set is build_sampler(algo, seed=3, length=64)


def processor_factory():
    return clrs.get_processor_factory("triplet_gmpnn", use_ln=True, nb_triplet_fts=8, nb_heads=1)


def num_test(algo):
    """CLRS-30 test-set size: 32 x num_samples_multiplier (64 for quickselect / kmp_matcher)."""
    return 32 * clrs.CLRS_30_ALGS_SETTINGS[algo]["num_samples_multiplier"]


def _postprocess(algo, spec, it, randomize_pos=False, rng=None):
    if randomize_pos:
        it = clrs.process_random_pos(it, rng)
    if algo in PRED_AS_INPUT_ALGOS:
        spec, it = clrs.process_pred_as_input(spec, it)
    spec, it = clrs.process_permutations(spec, it, True)
    return spec, it


def dataset(algo, n, num, seed=TEST_SEED, random_pos=False, **sampler_kwargs):
    """(feedback, spec): `num` trajectories of size n from the official sampler, test-style (pos = i/n, default
    sampler kwargs: ER p=0.5 graphs; string length n with needle n//5), padded to the set's longest trajectory.
    random_pos=True: the training/validation-style randomised pos input (clrs.process_random_pos, order kept; as in
    CLRS-WEIGHT-PROG's common.py)."""
    sampler, spec = clrs.build_sampler(algo, seed=seed, num_samples=num, length=n, **sampler_kwargs)
    fb = sampler.next(None)
    spec, it = _postprocess(algo, spec, iter([fb]), random_pos, np.random.RandomState(seed + 7))
    return next(it), spec


def test_set(algo, n=64, num=None, seed=TEST_SEED, **sampler_kwargs):
    return dataset(algo, n, num or num_test(algo), seed, **sampler_kwargs)


def _dp_with(d, data):
    """copy of a clrs DataPoint (attrs class) with new data."""
    return type(d)(d.name, d.location, d.type_, data)


def slice_fb(fb, lo, hi):
    """instances lo:hi of a feedback (inputs/outputs batch axis 0, hints axis 1)."""
    def dp(d, ax):
        return _dp_with(d, data=d.data[lo:hi] if ax == 0 else d.data[:, lo:hi])
    f = fb.features
    feats = f._replace(inputs=tuple(dp(d, 0) for d in f.inputs), hints=tuple(dp(d, 1) for d in f.hints),
                       lengths=f.lengths[lo:hi])
    return fb._replace(features=feats, outputs=tuple(dp(d, 0) for d in fb.outputs))


def trim_time(fb):
    """drop padded time steps beyond the longest trajectory in this batch."""
    T = int(np.max(fb.features.lengths))
    f = fb.features
    return fb._replace(features=f._replace(hints=tuple(_dp_with(d, data=d.data[:T]) for d in f.hints)))


_MODELS = {}


def model_for(algo, spec, dummy_fb, debug=False):
    """BaselineModel with the paper configuration (one per algorithm; parameters are passed in, never trained here).
    debug=True: predict also returns the processor hidden states of every step."""
    if (algo, debug) not in _MODELS or _MODELS[(algo, debug)].nb_msg_passing_steps != MODEL_KW["nb_msg_passing_steps"]:
        _MODELS[(algo, debug)] = clrs.models.BaselineModel(spec=[spec], dummy_trajectory=[dummy_fb], debug=debug,
                                                           processor_factory=processor_factory(), **MODEL_KW)
    return _MODELS[(algo, debug)]


def multi_model(debug=False):
    """the generalist (multi-task) BaselineModel: one shared processor, encoders/decoders algo_<i>_ for
    i = ALGOS.index(algo) (paper's multi-task setup, here over the lane's 4 algorithms)."""
    key = ("__multi__", debug)
    if key not in _MODELS:
        fbs, specs = zip(*[dataset(a, 16, 4, seed=1) for a in ALGOS])
        _MODELS[key] = clrs.models.BaselineModel(spec=list(specs), dummy_trajectory=list(fbs), debug=debug,
                                                 processor_factory=processor_factory(), **MODEL_KW)
        _MODELS[key]._cwp_init = [f.features for f in fbs]
    return _MODELS[key]


def init_params_multi(seed=0):
    m = multi_model()
    m.init(m._cwp_init, seed)
    return jax.device_get(m.params)


def rollout(params, algo, fb, spec, seed=0, multi=False):
    """one batch, own hints fed back: (outputs hard-postprocessed, per-step raw hint predictions [list over steps
    1..T-1 of {name: array}], hidden states (T-1, B, N, 128) after each processor step)."""
    m = multi_model(debug=True) if multi else model_for(algo, spec, fb, debug=True)
    m.params = params
    o, h, hid = m.predict(jax.random.PRNGKey(seed), fb.features, return_hints=True,
                          algorithm_index=ALGOS.index(algo) if multi else None)
    return jax.device_get(o), jax.device_get(h), np.asarray(jax.device_get(hid))


def init_params(algo, seed=0, n=16):
    """the parameter tree of the model (Haiku init with the paper's initialisers); defines names and shapes."""
    fb, spec = dataset(algo, n, 4, seed=1)
    m = model_for(algo, spec, fb)
    m.init([fb.features], seed)
    return jax.device_get(m.params)


def flatten(params):
    return {f"{mod}/{name}": np.asarray(v) for mod, d in params.items() for name, v in d.items()}


def unflatten(flat):
    out = {}
    for k, v in flat.items():
        mod, name = k.rsplit("/", 1)
        out.setdefault(mod, {})[name] = jnp.asarray(np.asarray(v, dtype=np.float32))
    return out


def check_tree(flat, ref_flat):
    """problems with a flat param dict versus the reference tree (missing/extra names, shape mismatches, non-finite)."""
    probs = []
    for k in sorted(set(ref_flat) - set(flat)):
        probs.append(f"missing {k} {ref_flat[k].shape}")
    for k in sorted(set(flat) - set(ref_flat)):
        probs.append(f"unexpected {k}")
    for k in sorted(set(flat) & set(ref_flat)):
        if tuple(np.shape(flat[k])) != tuple(ref_flat[k].shape):
            probs.append(f"shape {k}: {np.shape(flat[k])} != {ref_flat[k].shape}")
        elif not np.all(np.isfinite(flat[k])):
            probs.append(f"non-finite {k}")
    return probs


def predict(params, algo, fb, spec, batch_size=32, return_hints=False, seed=0, multi=False):
    """model outputs (hard-postprocessed) and optionally per-step soft hint predictions, own hints fed back."""
    m = multi_model() if multi else model_for(algo, spec, fb)
    m.params = params
    n_inst = fb.outputs[0].data.shape[0]
    outs, hints = [], []
    key = jax.random.PRNGKey(seed)
    for lo in range(0, n_inst, batch_size):
        b = slice_fb(fb, lo, min(n_inst, lo + batch_size))
        key, sub = jax.random.split(key)
        o, h = m.predict(sub, b.features, return_hints=return_hints,
                         algorithm_index=ALGOS.index(algo) if multi else None)
        outs.append(jax.device_get(o))
        if return_hints:
            hints.append(jax.device_get(h))
    outs = {k: _cat([o[k] for o in outs]) for k in outs[0]}
    return outs, hints


def _cat(dps):
    return _dp_with(dps[0], data=np.concatenate([np.asarray(d.data) for d in dps], 0))


def per_instance_exact(outputs, preds):
    """bool array: instance i has every output fully correct."""
    outputs = _ev._reduce_permutations_tuple(outputs)
    preds = _ev._reduce_permutations_dict(preds)
    n = outputs[0].data.shape[0]
    ok = np.ones(n, bool)
    for t in outputs:
        p = preds[t.name]
        for i in range(n):
            ok[i] &= bool(_ev._EVAL_FN[t.type_](np.asarray(p.data)[i:i + 1], t.data[i:i + 1]) == 1.0)
    return ok


def evaluate(params, algo, n=64, num=None, seed=TEST_SEED, batch_size=32, fb_spec=None, hints=False, multi=False,
             sampler_kwargs=None):
    """paper metric (CLRS score = mean over outputs of micro-F1/accuracy over the whole set), exact-match rate, and
    (hints=True) per-hint accuracy of the model's own re-fed hints against the ground-truth trajectory."""
    t0 = time.time()
    fb, spec = fb_spec or test_set(algo, n, num, seed, **(sampler_kwargs or {}))
    preds, hint_preds = predict(params, algo, fb, spec, batch_size, return_hints=hints, multi=multi)
    sc = clrs.evaluate(fb.outputs, preds)
    ex = per_instance_exact(fb.outputs, preds)
    res = {"algo": algo, "n": n, "num": int(len(ex)), "seed": seed, "sampler_kwargs": sampler_kwargs or {},
           "score": float(sc["score"]),
           "per_output": {k: float(v) for k, v in sc.items() if k != "score"}, "exact": float(ex.mean()),
           "n_exact": int(ex.sum()), "max_len": int(np.max(fb.features.lengths)), "s": round(time.time() - t0, 1)}
    if hints:
        res["hints"] = hint_accuracy(fb, spec, hint_preds, batch_size)
        tr = hint_trajectory(fb, spec, hint_preds, batch_size)
        res["hint_step_acc"] = tr["per_hint"]
        res["traj_exact"] = tr["traj_exact"]
        res["exact_and_traj"] = float(np.mean(tr["inst_ok"] & ex))
    return res


SCALAR_TOL = 2.5e-4    # a scalar hint matches if |pred - truth| <= SCALAR_TOL * max(1, |truth|)


def _hint_match(dp_true, dp_pred, t):
    """(B, ...) bool per element: postprocessed (hard) prediction equals the truth at time t."""
    truth = np.asarray(dp_true.data[t])
    pred = np.asarray(dp_pred.data)
    ty = dp_true.type_
    if ty in (_specs.Type.MASK_ONE, _specs.Type.CATEGORICAL):
        ok = np.argmax(pred, -1) == np.argmax(truth, -1)
        valid = np.all(truth != _specs.OutputClass.MASKED, -1)
        return ok | ~valid
    if ty == _specs.Type.MASK:
        return (truth == _specs.OutputClass.MASKED) | ((pred > 0.5) == (truth > 0.5))
    if ty == _specs.Type.SCALAR:
        return np.abs(pred - truth) <= SCALAR_TOL * np.maximum(1.0, np.abs(truth))
    if ty == _specs.Type.POINTER:
        return pred == truth
    raise ValueError(ty)


def hint_trajectory(fb, spec, hint_preds, batch_size):
    """strict track: per hint, the share of (instance, step) pairs whose hard-postprocessed own prediction equals the
    reference hint at that step (steps 1..length-1); per instance, whether every hint matched at every step."""
    n_inst = fb.outputs[0].data.shape[0]
    hits, tot, inst_ok = {}, {}, []
    for bi, lo in enumerate(range(0, n_inst, batch_size)):
        b = slice_fb(fb, lo, min(n_inst, lo + batch_size))
        truths = _ev._reduce_permutations_tuple(b.features.hints)
        lengths = np.asarray(b.features.lengths).astype(int)
        ok_b = np.ones(len(lengths), bool)
        for i, h in enumerate(hint_preds[bi]):
            hp = _ev._reduce_permutations_dict(
                decoders.postprocess(spec, h, sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True))
            t = i + 1
            live = t < lengths
            if not live.any():
                break
            for dt in truths:
                m = _hint_match(dt, hp[dt.name], t)
                m_inst = m.reshape(len(lengths), -1).all(-1)
                hits[dt.name] = hits.get(dt.name, 0) + int((m_inst & live).sum())
                tot[dt.name] = tot.get(dt.name, 0) + int(live.sum())
                ok_b &= m_inst | ~live
        inst_ok.append(ok_b)
    inst_ok = np.concatenate(inst_ok)
    return {"per_hint": {k: round(hits[k] / max(1, tot[k]), 5) for k in hits}, "traj_exact": float(inst_ok.mean()),
            "inst_ok": inst_ok}


def hint_accuracy(fb, spec, hint_preds, batch_size):
    """mean accuracy per hint (CLRS evaluate_hints over the batches: own-hint rollouts vs ground truth)."""
    acc = {}
    n_inst = fb.outputs[0].data.shape[0]
    for bi, lo in enumerate(range(0, n_inst, batch_size)):
        b = slice_fb(fb, lo, min(n_inst, lo + batch_size))
        hp = [decoders.postprocess(spec, h, sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True)
              for h in hint_preds[bi]]
        r = clrs.evaluate_hints(b.features.hints, b.features.lengths, hp)
        for k, v in r.items():
            acc.setdefault(k, []).append(float(np.mean(v)) if np.size(v) else float("nan"))
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
