"""E0061 (data): held-out long-context Tier-2 run at 128 GPUs (D28).

E0060 (P1:D1) showed decode surrogates calibrated on short contexts are ~8 %
optimistic on agent decode (contexts 2-20k); its Tier-2 run (seed 60) is the
calibration data. This is the held-out run and a 128-GPU Tier-2 check of the
agent workload: the agent optimum without retention (E0058: H200 P5 PP4xEP4
mc8 rank dispatch + D3 EP16 b64), 150 sessions x 8 turns at 2.5 sessions/s
(~70 % of its 3.55 capacity), seed 61; Tier-1 on the same workload for the
comparison. Tier-2 result.jsonl kept under tier2/.

    ./ie.sh experiments/E0061_longctx_decode_calibration/heldout.py
"""
import json
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
DEP = Deployment(hardware="h200", prefill_instances=5, decode_instances=3, prefill_ep=4, prefill_pp=4,
                 prefill_budget=2048, prefill_moe_chunks=8, prefill_dispatch="ranks",
                 decode_budget=64, decode_max_batch=64)
ROWS = workloads.agent_sessions(150, 8, 2.5, seed=61)
SLO = {"ttft_s": 5.0, "tpot_s": 0.05}


def run(tier):
    out = HERE / "tier2" / "agent128-seed61.jsonl"
    with tempfile.TemporaryDirectory(prefix="e0061-") as tmp:
        tmp = Path(tmp)
        res = out if tier == 2 and out.exists() else tmp / "r.jsonl"
        if res != out or not out.exists():
            cfg = tier2_config(DEP) if tier == 2 else te.tier1_config(tier2_config(DEP), models_for(DEP))
            te.write_config(dict(cfg, system={"kind": "agent_dag", "inner": cfg["system"]}), tmp / "c.toml")
            wl = workloads.write(ROWS, tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(res)], check=True, capture_output=True)
            if tier == 2:
                out.parent.mkdir(exist_ok=True)
                out.write_text(res.read_text())
                res = out
        m = steady.metrics(res, DEP.gpus, SLO, sides=(DEP.prefill_instances * DEP.width("prefill"),
                                                      DEP.decode_instances * DEP.width("decode")))
    return {k: m[k] for k in ("jct_s", "ttft_ms", "tpot_ms", "tokens_per_s_gpu", "slo_attainment", "queue")}


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as pool:
        t1, t2 = pool.map(run, (1, 2))
    rel = {"jct_p50": t1["jct_s"]["p50"] / t2["jct_s"]["p50"] - 1, "jct_p99": t1["jct_s"]["p99"] / t2["jct_s"]["p99"] - 1,
           "ttft_p50": t1["ttft_ms"]["p50"] / t2["ttft_ms"]["p50"] - 1, "ttft_p99": t1["ttft_ms"]["p99"] / t2["ttft_ms"]["p99"] - 1,
           "tpot_p50": t1["tpot_ms"]["p50"] / t2["tpot_ms"]["p50"] - 1,
           "tokens_per_s_gpu": t1["tokens_per_s_gpu"] / t2["tokens_per_s_gpu"] - 1}
    print("128-GPU agent, Tier-1 vs Tier-2", json.dumps({k: f"{v:+.1%}" for k, v in rel.items()}))
    cmv = cost_model_version()
    (HERE / f"heldout-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0061 held-out (128 GPUs)", "cost_model_version": cmv, "source_identity": source_identity(),
         "deployment": DEP.key, "workload": "agent_sessions(150, 8, 2.5, seed=61)", "rel_err": rel,
         "tier1": t1, "tier2": t2}, indent=1))
