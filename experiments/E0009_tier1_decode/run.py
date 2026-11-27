"""E0009: Tier-1 decode StepTime surrogate, H200 TP4/EP16, peak profile.

1. Fit on exported-graph lower bounds of 120 random decode batches
   (1-32 requests, kv 1-8192); hold out 30 more for a bound-level test.
2. Validate against Tier-2 engine decode rounds reconstructed from the frozen
   workload run at the same engine revision (E0008 result-c623652.jsonl).
3. Report per-query cost.

    ./ie.sh experiments/E0009_tier1_decode/run.py
"""
import json, statistics, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from adapters import tier1
from adapters.engines import ROOT, source_identity
from validation.decode_rounds import reconstruct

here = Path(__file__).parent
t0 = time.perf_counter()
model = tier1.fit("h200", 4, 16, "d", samples=120, seed=0)
fit_s = time.perf_counter() - t0
template = tier1._template("h200", 4, 16, "d", None)
held = tier1.sample_batches("d", 30, 1, 32, 8192, 1)
held_err = [abs(model.step_time(b) - tier1.bound_of(template, b, "h200")) / tier1.bound_of(template, b, "h200")
            for b in held]
rounds = reconstruct(ROOT / "experiments" / "E0008_tier1_feasibility" / "result-c623652.jsonl")
steady = [r for r in rounds if r.duration_s and not r.first_decode]
eng_err = []
t1 = time.perf_counter()
for r in steady:
    pred = model.step_time([(q, 1, kv) for q, kv in zip(r.requests, r.kv_lens)])
    eng_err.append((pred - r.duration_s) / r.duration_s)
query_us = (time.perf_counter() - t1) / max(1, len(steady)) * 1e6
abs_e = sorted(abs(e) for e in eng_err)
out = {
    "model": asdict(model), "feature_names": tier1.FEATURE_NAMES, "fit_host_s": fit_s,
    "heldout_bound": {"n": len(held_err), "median_rel_err": statistics.median(held_err),
                      "max_rel_err": max(held_err)},
    "engine_rounds": {"n": len(eng_err), "median_abs_rel_err": abs_e[len(abs_e) // 2],
                      "p90_abs_rel_err": abs_e[int(0.9 * (len(abs_e) - 1))],
                      "mean_signed_rel_err": statistics.fmean(eng_err)},
    "query_us": query_us, "source_identity": source_identity(),
}
print(json.dumps({k: v for k, v in out.items() if k != "source_identity"}, indent=1))
(here / f"result-{model.key['cost_model_version'][:12]}.json").write_text(json.dumps(out, indent=1))
