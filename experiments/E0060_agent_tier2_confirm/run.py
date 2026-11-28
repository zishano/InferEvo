"""E0060: per-instance Tier-2 confirmation of the agent-workload model (E0058).

E0015 (Phase 2, old lowering) found Tier-1 within 1 % of Tier-2 on mean agent
JCT but up to 120 % off on per-call TTFT tails. The agent conclusions
(E0058/E0059) are Tier-1, so the agent optimum's per-instance configuration
is re-checked as E0052 did for 8K/1K: P1:D1 (32 GPUs, H200), prefill
PP4xEP4 mc8 rank dispatch b2048, decode EP16 TP4 b64; 60 coding-agent sessions
x 8 turns (context 2k -> ~20k) at 0.5 sessions/s through agent_dag; Tier-2 vs
Tier-1 on the identical workload. No retention (Tier-2 shared execution has
no prefix caching). Tier-2 result.jsonl kept under tier2/.

    ./ie.sh experiments/E0060_agent_tier2_confirm/run.py
"""
import json
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters import steady, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import Deployment, tier2_config
from agent.controller import models_for

HERE = Path(__file__).parent
SLO = {"ttft_s": 5.0, "tpot_s": 0.05}
DEP = Deployment(hardware="h200", prefill_ep=4, prefill_pp=4, prefill_budget=2048, prefill_moe_chunks=8,
                 prefill_dispatch="ranks", decode_budget=64, decode_max_batch=64)
ROWS = workloads.agent_sessions(60, 8, 0.5, seed=60)


def run(tier):
    out = HERE / "tier2" / "agent.jsonl"
    with tempfile.TemporaryDirectory(prefix="e0060-") as tmp:
        tmp = Path(tmp)
        res = out if tier == 2 and out.exists() else tmp / "r.jsonl"
        if res != out or not out.exists():
            cfg = tier2_config(DEP) if tier == 2 else te.tier1_config(tier2_config(DEP), models_for(DEP))
            cfg = dict(cfg, system={"kind": "agent_dag", "inner": cfg["system"]})
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(ROWS, tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(res)], check=True, capture_output=True)
            if tier == 2:
                out.parent.mkdir(exist_ok=True)
                out.write_text(res.read_text())
                res = out
        calls = steady.calls(res)
        m = steady.metrics(res, DEP.gpus, SLO, warm=0, cool=0)
    ttft = sorted(c["ttft"] for c in calls)
    return {"calls": len(calls), "jct_s": m["jct_s"], "ttft_ms": {p: ttft[min(len(ttft) - 1, int(q * len(ttft)))] * 1e3
            for p, q in (("p50", .5), ("p90", .9), ("p99", .99))}, "tpot_ms": m["tpot_ms"],
            "ttft_by_call": [c["ttft"] for c in sorted(calls, key=lambda c: (c["request"], c["arrival"]))]}


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as pool:
        t1, t2 = pool.map(run, (1, 2))
    pair = [(a, b) for a, b in zip(t1["ttft_by_call"], t2["ttft_by_call"]) if b > 0]
    per_call = sorted(abs(a / b - 1) for a, b in pair)
    rel = {"jct_p50": t1["jct_s"]["p50"] / t2["jct_s"]["p50"] - 1, "jct_p99": t1["jct_s"]["p99"] / t2["jct_s"]["p99"] - 1,
           **{f"ttft_{p}": t1["ttft_ms"][p] / t2["ttft_ms"][p] - 1 for p in ("p50", "p90", "p99")},
           "tpot_p50": t1["tpot_ms"]["p50"] / t2["tpot_ms"]["p50"] - 1,
           "per_call_ttft_abs_err_p50": statistics.median(per_call), "per_call_ttft_abs_err_p90": per_call[int(.9 * len(per_call))]}
    print("Tier-1 vs Tier-2", json.dumps({k: f"{v:+.1%}" for k, v in rel.items()}))
    print("Tier-2", json.dumps({"jct_s": t2["jct_s"], "ttft_ms": t2["ttft_ms"], "tpot_ms": t2["tpot_ms"]}))
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0060", "cost_model_version": cmv, "source_identity": source_identity(), "deployment": DEP.key,
         "workload": "agent_sessions(60, 8, 0.5, seed=60)", "rel_err": rel,
         "tier1": {k: v for k, v in t1.items() if k != "ttft_by_call"},
         "tier2": {k: v for k, v in t2.items() if k != "ttft_by_call"}}, indent=1))
