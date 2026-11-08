"""CLRS-GENERALIST: one shared processor, per-task encoders/decoders (the multi-task setting of Ibarz et al. 2022).

Program format (what build.py writes to OUTDIR/params.npz): flat float32 arrays keyed
  'P|<name>'      for every processor parameter (names under net/mpnn_aggr_clrs_processor/), ONE set for all tasks
  '<algo>|<name>' for that task's encoder/decoder parameters, named as in the single-task model (algo_0_...)
The single-task network of task `algo` is P + T[algo]. In CLRS's multi-task Net the encoders/decoders of task i are
named algo_<i>_... and only those of the selected algorithm_index are used, with the shared processor; so P + T[algo]
evaluated as a single-task model is the multi-task model at that index. multi_selfcheck() checks this claim against
the official multi-task BaselineModel.predict, every hard hint at every step plus outputs.
"""
import hashlib
import json
import re

import numpy as np
import jax
import jax.numpy as jnp

import common as C

PROC = "net/mpnn_aggr_clrs_processor/"
HEADLINE = ["activity_selector", "articulation_points", "bellman_ford", "bfs", "bridges", "bubble_sort",
            "dag_shortest_paths", "dfs", "dijkstra", "graham_scan", "heapsort", "insertion_sort", "jarvis_march",
            "kmp_matcher", "minimum", "mst_prim", "naive_string_matcher", "quicksort",
            "strongly_connected_components", "task_scheduling", "topological_sort"]
_REF = {}


def ref_flat(algo):
    if algo not in _REF:
        _REF[algo] = C.flatten(C.init_params(algo))
    return _REF[algo]


def proc_shapes(algo="bfs"):
    return {k: tuple(v.shape) for k, v in ref_flat(algo).items() if k.startswith(PROC)}


def task_shapes(algo):
    return {k: tuple(v.shape) for k, v in ref_flat(algo).items() if not k.startswith(PROC)}


def split_single(flat):
    """a single-task flat dict -> (processor dict, task enc/dec dict)."""
    P = {k: np.asarray(v, np.float32) for k, v in flat.items() if k.startswith(PROC)}
    T = {k: np.asarray(v, np.float32) for k, v in flat.items() if not k.startswith(PROC)}
    return P, T


def to_npz_dict(P, T):
    d = {"P|" + k: np.asarray(v, np.float32) for k, v in P.items()}
    for a, Ta in T.items():
        d.update({a + "|" + k: np.asarray(v, np.float32) for k, v in Ta.items()})
    return d


def from_npz_dict(d):
    P, T = {}, {}
    for key, v in d.items():
        pre, name = key.split("|", 1)
        if pre == "P":
            P[name] = np.asarray(v, np.float32)
        else:
            T.setdefault(pre, {})[name] = np.asarray(v, np.float32)
    return P, T


def load(path):
    return from_npz_dict(dict(np.load(path)))


def save(path, P, T):
    np.savez(path, **to_npz_dict(P, T))


def _sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        a = np.ascontiguousarray(np.asarray(d[k], np.float32))
        h.update(k.encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def proc_sha256(P):
    """hash of the shared processor parameter set (names, shapes, float32 bytes)."""
    return _sha(P)


def task_sha256(Ta):
    return _sha(Ta)


def check_program(P, T, algos=None):
    """problems: processor keys/shapes vs the reference processor, each task's keys/shapes vs its reference
    encoders/decoders, non-finite values, unknown task names."""
    probs = []
    algos = list(T) if algos is None else list(algos)
    for a in algos:
        if a not in T:
            probs.append(f"task {a} missing")
            continue
        try:
            ref = ref_flat(a)
        except Exception as e:  # noqa: BLE001
            probs.append(f"unknown task {a}: {e!r}"[:200])
            continue
        probs += [f"P: {p}" for p in C.check_tree(P, {k: v for k, v in ref.items() if k.startswith(PROC)})]
        probs += [f"{a}: {p}" for p in C.check_tree(T[a], {k: v for k, v in ref.items() if not k.startswith(PROC)})]
    return sorted(set(probs))


def assemble(P, T, algo):
    """the single-task flat dict P + T[algo] (checked)."""
    probs = check_program(P, {algo: T[algo]}, [algo])
    assert not probs, probs[:10]
    full = dict(P)
    full.update(T[algo])
    return full


def multi_selfcheck(P, T, algos=None, n=16, num=4, seed=4242, check=None):
    """official multi-task BaselineModel (spec = the program's tasks, one shared processor) vs our per-task
    streamed evaluation of P + T[algo]: every hard hint at every live step and every output, compared exactly."""
    import clrs
    import stream as SM
    algos = list(T) if algos is None else list(algos)
    fbs, specs = zip(*[C.dataset(a, 16, 4, seed=1) for a in algos])
    m = clrs.models.BaselineModel(spec=list(specs), dummy_trajectory=list(fbs), processor_factory=C.processor_factory(),
                                  **C.MODEL_KW)
    m.init([f.features for f in fbs], 0)
    ref = C.flatten(jax.device_get(m.params))
    flat = dict(P)
    for i, a in enumerate(algos):
        for k, v in T[a].items():
            k2 = k.replace("/algo_0_", f"/algo_{i}_", 1)
            assert k2 in ref, k2
            flat[k2] = v
    probs = C.check_tree(flat, ref)
    assert not probs, probs[:10]
    m.params = C.unflatten(flat)
    out = []
    for i, a in enumerate(algos):
        if check is not None and a not in check:
            continue
        fb, spec = C.dataset(a, n, num, seed)
        fb = C.trim_time(fb)
        o1, h1 = m.predict(jax.random.PRNGKey(0), fb.features, algorithm_index=i, return_hints=True)
        params = C.unflatten(assemble(P, T, a))
        first, step, outs = SM._fns(a, spec, fb)
        feats = jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
                                       SM._first_hint_features(fb))
        Tn = int(np.max(fb.features.lengths))
        same_h, n_cmp, st = True, 0, None
        for s in range(max(1, Tn - 1)):
            st, hard = first(params, feats) if s == 0 else step(params, feats, st, jnp.int32(s))
            r = SM._post_hints(spec, h1[s])
            hard = jax.device_get(hard)
            for k in r:
                same_h &= bool(np.array_equal(np.asarray(r[k]), np.asarray(hard[k])))
                n_cmp += 1
        o2 = jax.device_get(outs(st))
        same_o = all(np.array_equal(np.asarray(o1[k].data), np.asarray(o2[k].data)) for k in o1)
        out.append({"algo": a, "algorithm_index": i, "n": n, "num": num, "seed": seed, "hints_identical": bool(same_h),
                    "outputs_identical": bool(same_o), "hint_arrays_compared": n_cmp})
    return out


def single_names_ok(name):
    return bool(re.match(r"^net/~_construct_encoders_decoders/algo_0_", name))
