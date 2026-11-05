"""Rebuild every network from its program in build/ and compare with the shipped weights.

    python make_weights.py            run each build/NAME.py in a fresh process, check sha256 against weights/SHA256SUMS
    python make_weights.py --write    also (re)write weights/NAME.npz and weights/SHA256SUMS

Each build program writes config.json and one .npy per tensor into the directory given as its argument; this script
packs that output with arch.save().
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import arch

ROOT = os.path.dirname(os.path.abspath(__file__))
NETS = ("two_layer", "one_layer", "looped")
SUMS = os.path.join(ROOT, "weights", "SHA256SUMS")


def rebuild(name, out_path):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "build", name + ".py"), d], check=True, cwd=d)
        cfg = json.load(open(os.path.join(d, "config.json")))
        W = {}
        for k in arch.shapes(cfg):
            v = np.load(os.path.join(d, k + ".npy"))
            assert np.array_equal(v.astype(np.float32), v), f"{name}: {k} is not exactly float32"
            W[k] = v.astype(np.float32)
        arch.save(out_path, cfg, W)
    return hashlib.sha256(open(out_path, "rb").read()).hexdigest()


def main():
    write = "--write" in sys.argv[1:]
    shipped = {}
    if os.path.exists(SUMS):
        shipped = {line.split()[1]: line.split()[0] for line in open(SUMS) if line.strip()}
    sums, ok = {}, True
    with tempfile.TemporaryDirectory() as tmp:
        for name in NETS:
            fn = name + ".npz"
            h = rebuild(name, os.path.join(ROOT, "weights", fn) if write else os.path.join(tmp, fn))
            sums[fn] = h
            same = shipped.get(fn) == h
            ok &= same or write
            print(f"{name:10s} sha256 {h}  {'matches' if same else 'DIFFERS from'} weights/SHA256SUMS")
    if write:
        open(SUMS, "w").write("".join(f"{sums[f]}  {f}\n" for f in sorted(sums)))
        print("wrote weights/*.npz and weights/SHA256SUMS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
