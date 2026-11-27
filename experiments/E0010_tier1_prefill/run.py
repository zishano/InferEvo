"""E0010: Tier-1 prefill StepTime surrogate (chunked prefill, 128-token budget).

Fit on exported-graph bounds of random prefill batches (1-8 requests, total
<= 128 tokens, kv 0 or 1-8192) for H200 TP4 and B300 TP4 (peak); held-out
bound error; check against E0004 engine prefill rounds (single 5-token request).

    ./ie.sh experiments/E0010_tier1_prefill/run.py
"""
import glob, json, statistics, time
from dataclasses import asdict
from pathlib import Path
from adapters import tier1
from adapters.engines import ROOT, source_identity

here = Path(__file__).parent
e4 = json.loads(open(sorted(glob.glob(str(ROOT / "experiments/E0004_kimi_round_differential/result-*.json")))[-1]).read())
out = {"cases": [], "source_identity": source_identity()}
for hw, tp in (("h200", 4), ("b300", 4)):
    t0 = time.perf_counter()
    model = tier1.fit(hw, tp, 16, "p", samples=80, seed=0, max_requests=8, max_kv=8192, max_q=128)
    fit_s = time.perf_counter() - t0
    template = tier1._template(hw, tp, 16, "p", None)
    held = tier1.sample_batches("p", 20, 1, 8, 8192, 128)
    errs = []
    for b in held:
        y = tier1.bound_of(template, b, hw)
        errs.append(abs(model.step_time(b) - y) / y)
    engine = [r for r in e4["rows"] if r["hardware"] == hw and r["tp"] == tp and r["instance"] == "p0"]
    eng = [(model.step_time([(0, 5, 0)]) - r["engine_ms"] / 1e3) / (r["engine_ms"] / 1e3) for r in engine]
    case = {"hardware": hw, "tp": tp, "model": asdict(model), "fit_host_s": fit_s,
            "heldout_median_rel_err": statistics.median(errs), "heldout_max_rel_err": max(errs),
            "e0004_engine_rel_err": eng}
    out["cases"].append(case)
    print(f"{hw} tp{tp}: train median {model.train_median_rel_err:.4f} p90 {model.train_p90_rel_err:.4f} | "
          f"held-out median {case['heldout_median_rel_err']:.4f} max {case['heldout_max_rel_err']:.4f} | "
          f"E0004 engine {eng} | fit {fit_s:.0f}s", flush=True)
cmv = out["cases"][0]["model"]["key"]["cost_model_version"]
(here / f"result-{cmv[:12]}.json").write_text(json.dumps(out, indent=1))
