"""specs/mk_cases.py: write the frozen case lists for Experiments 5 and 6 (PROTOCOL.md) into specs/."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
NETS5 = ("anylen", "two", "ng1", "ng2", "tf")
ADAM5 = (1e-6, 1e-5, 1e-4, 1e-3)
OWN = dict(ng1=dict(opt="adam", lr=5e-4, adam_eps=1e-3, clip=4.0, ls=0.1, dropout=True, batch=32),
           ng2=dict(opt="adam", lr=5e-4, adam_eps=1e-3, clip=4.0, ls=0.1, dropout=True, batch=32),
           tf=dict(opt="adam", lr=1e-3, clip=1.0, batch=64))
SGD6 = [10.0 ** e for e in range(-12, -3)]
ADAM6 = [10.0 ** e for e in range(-10, -2)]

scan5 = [dict(name=f"s5-{n}-{c}-{s}", net=n, cond=c, grad_seed=7700 + s) for n in NETS5 for c in ("low", "high") for s in range(8)]
train5 = [dict(name=f"t5-{n}-{c}-adam-{lr:g}-{s}", net=n, cond=c, opt="adam", lr=lr, seed=s)
          for n in NETS5 for c in ("low", "high") for lr in ADAM5 for s in (1, 2, 3)]
train5 += [dict(name=f"t5-{n}-{c}-own-{s}", net=n, cond=c, seed=s, **OWN[n]) for n in OWN for c in ("low", "high") for s in (1, 2, 3)]
scan6 = [dict(name=f"s6-untied-low-{s}", net="untied", cond="low", grad_seed=7700 + s) for s in range(8)]
train6 = [dict(name=f"t6-{n}-{o}-{lr:g}-{s}", net=n, cond="low", opt=o, lr=lr, seed=s)
          for n in ("anylen", "untied") for o, lrs in (("sgd", SGD6), ("adam", ADAM6)) for lr in lrs for s in (1, 2, 3)]
for name, cases in (("scan5", scan5), ("train5", train5), ("scan6", scan6), ("train6", train6)):
    (H / f"{name}.json").write_text(json.dumps(cases))
    print(name, len(cases))
# Split by machine: written and untied anylen on CPU, trained nets on the L4. Same cases, nothing added or removed.
W = ("anylen", "two")
for name, cases in (("scan5", scan5), ("train5", train5)):
    (H / f"{name}_w.json").write_text(json.dumps([c for c in cases if c["net"] in W]))
    (H / f"{name}_t.json").write_text(json.dumps([c for c in cases if c["net"] not in W]))
    print(name, "w", sum(c["net"] in W for c in cases), "t", sum(c["net"] not in W for c in cases))
# 2026-11-08 08:42 UTC: ng1 fails length 64 at the start (g-check), so its strict set drops 64 (PROTOCOL.md rule).
# The g-scan job had started with the old code; its ng1 cases are rerun in g-scan-ng1 and the old ng1 files are not used.
(H / "scan5_ng1.json").write_text(json.dumps([c for c in scan5 if c["net"] == "ng1"]))
