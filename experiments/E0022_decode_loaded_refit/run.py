"""E0022: refit the H200 TP4 decode surrogate on loaded Tier-2 rounds.

Rounds are reconstructed per decode instance from kept Tier-2 result.jsonl
(validation.decode_rounds.reconstruct_multi; busy rounds, first-decode
rounds excluded). Joint fit = loaded rounds + the E0015 graph-bound samples
(keeps coverage of large batch / long KV outside the loaded range) with the
bound samples' targets left as is. Cross-check: train on one run, test on
the other; end to end: Tier-1 runs of both points vs Tier-2 TPOT/TTFT.

    ./ie.sh experiments/E0022_decode_loaded_refit/run.py
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
old = tier1.Tier1Model(**json.loads((E / "E0015_agent_tier1/decode_model.json").read_text()))
runs = {b: E / f"E0018_controller_budget/tier2/h200-p1x4-d3x4-ep16-b{b}-32-m32-32.jsonl" for b in (128, 2048)}


def loaded(path):
    out = []
    for rs in reconstruct_multi(path).values():
        out += [([(i, 1, kv) for i, kv in zip(r.requests, r.kv_lens)], r.duration_s)
                for r in rs if r.duration_s is not None and not r.first_decode]
    return out


data = {b: loaded(p) for b, p in runs.items()}
bounds = tier1.sample_batches("d", 120, 0, 32, 8192, 512)       # E0015 training batches
bound_y = [old.step_time(b) for b in bounds]                    # old model == bound fit (0.8 % median)
key = dict(old.key, target="loaded Tier-2 decode rounds (E0018) + graph-bound samples (E0015)")


def err(model, rows):
    e = [(model.step_time(b) - y) / y for b, y in rows]
    return {"median_signed": statistics.median(e), "median_abs": statistics.median(map(abs, e)),
            "p90_abs": float(np.quantile(np.abs(e), .9))}


cross = {}
for train, test in ((128, 2048), (2048, 128)):
    rows = data[train]
    m = tier1.fit_samples([b for b, _ in rows] + bounds, [y for _, y in rows] + bound_y, key)
    cross[f"train{train}_test{test}"] = {"old": err(old, data[test]), "new": err(m, data[test])}
allrows = data[128] + data[2048]
new = tier1.fit_samples([b for b, _ in allrows] + bounds, [y for _, y in allrows] + bound_y, key)
print(json.dumps(cross, indent=1))
print("bound-sample error of new model", err(new, list(zip(bounds, bound_y))))

pre = tier1.Tier1Model(**json.loads((E / "E0017_prefill_budgets/result.json").read_text())["prefill"])
t2 = json.loads((E / "E0018_controller_budget/confirm.json").read_text())
e2e = {}
for b in (128, 2048):
    dep = Deployment(prefill_instances=1, decode_instances=3, prefill_budget=b)
    ref = t2[dep.key]["tier2"]
    e2e[dep.key] = {"tier2": ref}
    for name, dm in (("old", old), ("new", new)):
        cfg = te.tier1_config(tier2_config(dep), {"p": pre, "d": dm})
        with tempfile.TemporaryDirectory(prefix="e0022-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            te.run(tmp / "c.toml", workloads.write(workloads.scaled(4), tmp / "w.jsonl"), tmp / "r.jsonl")
            r = feedback.requests(tmp / "r.jsonl", dep.gpus)
        got = {"tokens_per_s_gpu": r["tokens_per_s_gpu"], "ttft_p99_ms": r["ttft_ms"]["p99"],
               "tpot_p50_ms": r["tpot_ms"]["p50"]}
        e2e[dep.key][name] = {k: (got[k] - ref[k]) / ref[k] for k in got}
        print(dep.key, name, {k: f"{v:+.1%}" for k, v in e2e[dep.key][name].items()})
(here / "decode_model.json").write_text(json.dumps(asdict(new)))
(here / "result.json").write_text(json.dumps({"experiment": "E0022", "cross": cross, "end_to_end": e2e,
    "n_loaded_rounds": {b: len(v) for b, v in data.items()}, "decode": asdict(new),
    "source_identity": source_identity()}, indent=1))
