"""Local smoke test (60 s, one thread): code paths run, untied equals tied at the start. Not a score."""
import sys, time, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j
torch.set_num_threads(1)
I = sys.argv[1]
t0 = time.time()
tied, un = j.make("anylen", I), j.make("untied", I)
for n, m in ((2, 3), (4, 4), (5, 2)):
    b = tied.batch(np.random.default_rng(1), n, m, 4)
    with torch.no_grad():
        d = float((tied.logits(tied.W0, b)[0] - un.logits(un.W0, b)[0]).abs().max())
    print("untied-vs-tied max |dlogit|", n, m, d)
print("keys shared", sorted(k for k in un.W0 if not k.startswith("c")), "copies", un.J, "params", sum(v.numel() for v in un.W0.values()))
g = j.grad_batch
j.grad_batch = lambda net, seed: net.batch(np.random.default_rng(seed), *net.grad_len, 4)
for name in ("anylen", "untied", "two", "ng1", "tf"):
    net = j.make(name, I)
    b = net.batch(np.random.default_rng(5), *net.grad_len, 4)
    ok = j.correct(net, net.W0, b)
    dirs, gn, cos, extra = j.directions(net, net.W0, 7700)
    print(name, "params", sum(v.numel() for v in net.W0.values()), "ok on 4", ok.tolist(), "|g|", round(gn, 4), "cos", cos,
          "dirs", sorted(dirs), {k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v[:3]]) for k, v in extra.items()}, round(time.time() - t0, 1))
    if name in ("ng1", "tf"):
        a = j.low_alpha(net)
        print("  low alpha on 4 problems", a)
    c = dict(net=name, cond="low", opt="adam", lr=1e-6, seed=1, steps=2, batch=4, **(dict(ls=0.1, dropout=True, clip=4.0, adam_eps=1e-3) if name == "ng1" else {}))
    j.strict = lambda net, W, upto=None: (True, None, 0.0, 1.0)   # skip the full strict set locally
    r = j.train(c, I)
    print("  train 2 steps ok, loss", r["loss_first"], r["loss_last"], round(time.time() - t0, 1))
    j.strict = j.__dict__["strict"]
print("total", round(time.time() - t0, 1))
