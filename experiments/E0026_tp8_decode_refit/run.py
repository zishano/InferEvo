"""E0026: refit the H200 TP8 decode surrogate on loaded Tier-2 rounds (E0022
method). One Tier-2 run (P1:D3 TP8 b2048) -> held-out by decode instance
(train on 2, test on the third, 3 folds). End to end vs Tier-2, old vs new.

    ./ie.sh experiments/E0026_tp8_decode_refit/run.py
"""
import json, statistics, tempfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
from adapters import feedback, tier1, tier1_engine as te, workloads
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from validation.decode_rounds import reconstruct_multi

here = Path(__file__).parent
E = here.parent
tp8 = json.loads((E / "E0020_tp8_surrogates/result.json").read_text())
old, pre = tier1.Tier1Model(**tp8["decode"]), tier1.Tier1Model(**tp8["prefill"])
dep = Deployment(prefill_instances=1, decode_instances=3, prefill_budget=2048, prefill_tp=8, decode_tp=8)
inst = {k: [([(i, 1, kv) for i, kv in zip(r.requests, r.kv_lens)], r.duration_s) for r in rs
            if r.duration_s is not None and not r.first_decode]
        for k, rs in reconstruct_multi(E / f"E0018_controller_budget/tier2/{dep.key}.jsonl").items()}
bounds = tier1.sample_batches("d", 120, 3, 32, 32768, 512)     # E0020 training batches
bound_y = [old.step_time(b) for b in bounds]
key = dict(old.key, target="loaded Tier-2 decode rounds (E0018 TP8) + graph-bound samples (E0020)")


def err(m, rows):
    e = [(m.step_time(b) - y) / y for b, y in rows]
    return {"median_signed": statistics.median(e), "median_abs": statistics.median(map(abs, e)),
            "p90_abs": float(np.quantile(np.abs(e), .9))}


def fit(rows):
    return tier1.fit_samples([b for b, _ in rows] + bounds, [y for _, y in rows] + bound_y, key)


folds = {}
for k in inst:
    train = [r for j, rs in inst.items() if j != k for r in rs]
    folds[k] = {"old": err(old, inst[k]), "new": err(fit(train), inst[k])}
    print("fold", k, {a: {b: round(c, 4) for b, c in v.items()} for a, v in folds[k].items()})
new = fit([r for rs in inst.values() for r in rs])
ref = json.loads((E / "E0018_controller_budget/confirm.json").read_text())[dep.key]["tier2"]
e2e = {}
for name, dm in (("old", old), ("new", new)):
    cfg = te.tier1_config(tier2_config(dep), {"p": pre, "d": dm})
    with tempfile.TemporaryDirectory(prefix="e0026-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        te.run(tmp / "c.toml", workloads.write(workloads.scaled(4), tmp / "w.jsonl"), tmp / "r.jsonl")
        r = feedback.requests(tmp / "r.jsonl", dep.gpus, {"ttft_s": 2.0, "tpot_s": 0.05})
    got = {"tokens_per_s_gpu": r["tokens_per_s_gpu"], "ttft_p99_ms": r["ttft_ms"]["p99"], "tpot_p50_ms": r["tpot_ms"]["p50"]}
    e2e[name] = {k: (got[k] - ref[k]) / ref[k] for k in got} | {"goodput": r["goodput_tokens_per_s_gpu"]}
    print(name, {k: (f"{v:+.1%}" if k != "goodput" else round(v, 2)) for k, v in e2e[name].items()})
(here / "decode_model.json").write_text(json.dumps(asdict(new)))
(here / "result.json").write_text(json.dumps({"experiment": "E0026", "folds": folds, "end_to_end": e2e,
    "decode": asdict(new), "source_identity": source_identity()}, indent=1))
