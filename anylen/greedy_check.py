"""Free-running check (local, one thread, < 60 s): decode with the network's own digits, not the teacher's. Start from
BOS a * b EOS, run the halting rule, append the argmax at the last position, repeat for n + m digits (the caller counts
them: the network has no end-of-answer token). Then two digits past the end, to record what it emits there."""
import json

import numpy as np
import torch

import arch

torch.set_num_threads(1)
torch.set_grad_enabled(False)
cfg, W = arch.load("weights/anylen.npz")
rng = np.random.default_rng(20261105)
rows = []
for n, m, cnt in ((1, 1, 20), (3, 2, 20), (2, 7, 10), (7, 3, 10), (5, 5, 10), (12, 1, 5), (10, 10, 3)):
    A, B = rng.integers(0, 10, (cnt, n)), rng.integers(0, 10, (cnt, m))
    t, C = arch.encode(A, B)
    seq = t[:, :n + m + 3]
    got, busy_last = [], 0
    for k in range(n + m + 2):
        lg, halt = arch.forward(W, seq, cfg)
        d = lg[:, -1].argmax(-1)
        busy_last += int((d == arch.BUSY).sum())
        got.append(d)
        seq = torch.cat([seq, d[:, None]], 1)
    g = torch.stack(got, 1).numpy()
    rows.append(dict(n=n, m=m, problems=cnt, wrong=int((g[:, :n + m] != C).any(1).sum()), busy_at_read=busy_last,
                     past_end_digits=sorted({int(x) for x in g[:, n + m:].ravel()})))
    print(rows[-1], flush=True)
json.dump(rows, open("results/greedy-local.json", "w"), indent=1)
