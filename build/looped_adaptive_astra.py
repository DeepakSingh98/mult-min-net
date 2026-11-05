"""Adaptive multiplication circuit; explicit edits to the supplied hand-built program.
(GPT-6 Astra's program, unchanged except the path to looped.py, which was /work/prev/build/looped.py in its sandbox.)
No training, fitting, search, or saved-weight inputs.
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

CFG = dict(n_layer=1, loop=42, n_head=2, d_head=7, d_model=50, d_mlp=44)

def build():
    spec = importlib.util.spec_from_file_location(
        "previous_looped", Path(__file__).resolve().parent / "looped.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    W, layout = previous.build()
    W["h0.mlp.w2"][41, layout["token_balance"]] = -1/32
    W["h0.mlp.b1"][41] = 32*0.75
    W["lnf.w"][0:2] *= 0.5
    W["h0.ln2.b"][layout["counter"]] = -5
    W["h0.mlp.b1"][43] = 32*5
    return W

def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out/"config.json").write_text(json.dumps(CFG, indent=2))
    for name, value in build().items():
        np.save(out/(name+".npy"), np.asarray(value, dtype=np.float32))

if __name__ == "__main__":
    main()
