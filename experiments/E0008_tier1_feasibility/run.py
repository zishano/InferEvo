"""E0008: is the exported-graph lower bound a usable Tier-1 core for
multi-request decode rounds?

Engine rounds are reconstructed from a Tier-2 run of the frozen workload
(H200 TP4/EP16, peak profile; result.jsonl stored here). For sampled steady
rounds the exact batch is exported (same request ids, kv lengths, routing
seed 0) and bounded with validation.round_bounds; we report engine/LB and the
export host time.

    ./ie.sh experiments/E0008_tier1_feasibility/run.py
"""
import json, random, statistics, time
from pathlib import Path
from adapters.cost_model import cost_model_version
from adapters.engines import enable_bridge_imports, source_identity
from validation.decode_rounds import reconstruct
from validation.round_bounds import round_bounds

enable_bridge_imports()
from rooflang_bridge.kimi_k3_shared import KimiTemplate  # noqa: E402

here = Path(__file__).parent
rounds = reconstruct(here / "result.jsonl")
steady = [r for r in rounds if r.duration_s is not None and not r.first_decode]
handoff = [r for r in rounds if r.duration_s is not None and r.first_decode]
random.seed(0)
by_size = {}
for r in steady:
    by_size.setdefault(len(r.requests), []).append(r)
sample = [x for size in sorted(by_size) for x in random.sample(by_size[size], min(8, len(by_size[size])))]
template = KimiTemplate(4, 16, {"gpu_model": "h200", "gpu_count": 16}, 0, stage="d")
rows = []
for r in sample:
    t0 = time.perf_counter()
    ops = template.export([{"q_len": 1, "kv_len": k, "lm_head_len": 1} for k in r.kv_lens],
                          r.requests, details=False)["operations"]
    export_s = time.perf_counter() - t0
    b = round_bounds(ops, "h200")
    rows.append({"batch": len(r.requests), "sum_kv": sum(r.kv_lens), "engine_ms": r.duration_s * 1e3,
                 "lower_ms": b.lower_s * 1e3, "ratio": r.duration_s / b.lower_s,
                 "operations": b.operations, "export_s": export_s})
    print(f"n={len(r.requests)} kv={sum(r.kv_lens):5d} engine {r.duration_s*1e3:6.3f} ms "
          f"LB {b.lower_s*1e3:6.3f} ms ratio {r.duration_s/b.lower_s:.3f} export {export_s:.2f}s", flush=True)
summary = {}
for n in sorted({x["batch"] for x in rows}):
    rs = [x["ratio"] for x in rows if x["batch"] == n]
    summary[n] = {"n": len(rs), "ratio_min": min(rs), "ratio_median": statistics.median(rs), "ratio_max": max(rs)}
print(json.dumps(summary, indent=1))
print("rounds: steady", len(steady), "handoff", len(handoff), "idle-start", sum(r.duration_s is None for r in rounds))
cmv = cost_model_version()
(here / f"result-{cmv[:12]}.json").write_text(json.dumps({
    "experiment": "E0008", "cost_model_version": cmv, "source_identity": source_identity(),
    "tier2_run": "run_kimi_k3_shared_workload.sh (H200 TP4, peak), result.jsonl in this dir",
    "rows": rows, "summary": summary,
    "counts": {"steady": len(steady), "handoff": len(handoff)}}, indent=1))
