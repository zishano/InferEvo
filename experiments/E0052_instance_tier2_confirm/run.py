"""E0052: Tier-2 confirmation of the current optima per instance (replaces E0046).

Tier-1 and Tier-2 share scheduler, router, KV and P/D handoff (D19); only the
round time differs. So the 128-GPU optima are checked on one instance pair
(P1:D1, 32 GPUs) carrying the optimum's per-instance configuration under a
load near its operating point, Tier-2 vs Tier-1 on the identical workload:
  official: prefill PP4xEP4 b2048 mc8 rank dispatch | decode EP16 TP4 b64
  fp8:      same + linear fp8 both sides (M-FP8LINEAR, quality UNVERIFIED)
Rates: decode is the P1:D1 limit (~13.0/5 = 2.6 req/s official, ~15/5 = 3.0
fp8 per decode instance at 128 GPUs) -> 2.4 and 2.8 req/s; 240 requests 8K/1K.
Tier-2 result.jsonl kept under tier2/.

    ./ie.sh experiments/E0052_instance_tier2_confirm/run.py
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
SLO = {"ttft_s": 5.0, "tpot_s": 0.05}
BASE = dict(hardware="h200", prefill_ep=4, prefill_pp=4, prefill_budget=2048, prefill_moe_chunks=8,
            prefill_dispatch="ranks", decode_budget=64, decode_max_batch=64)
CASES = {"official": (Deployment(**BASE), (240, 2.4, 8192, 1024, 52)),
         "fp8": (Deployment(**BASE, prefill_linear="fp8", decode_linear="fp8"), (240, 2.8, 8192, 1024, 52))}


def run(name, tier):
    dep, wl_spec = CASES[name]
    out = HERE / "tier2" / f"{name}.jsonl"
    with tempfile.TemporaryDirectory(prefix="e0052-") as tmp:
        tmp = Path(tmp)
        res = out if tier == 2 and out.exists() else tmp / "r.jsonl"
        if res != out or not out.exists():
            cfg = tier2_config(dep) if tier == 2 else te.tier1_config(tier2_config(dep), models_for(dep))
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(workloads.fixed(*wl_spec[:4], seed=wl_spec[4]), tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(res)], check=True, capture_output=True)
            if tier == 2:
                out.parent.mkdir(exist_ok=True)
                out.write_text(res.read_text())
                res = out
        m = steady.metrics(res, dep.gpus, SLO, sides=(dep.width("prefill"), dep.width("decode")))
    keep = ("tokens_per_s_gpu", "prefill_tokens_per_s_gpu", "decode_tokens_per_s_gpu", "slo_attainment", "queue")
    return {**{k: m[k] for k in keep}, "ttft_ms": m["ttft_ms"], "tpot_ms": m["tpot_ms"]}


if __name__ == "__main__":
    jobs = [(n, t) for n in CASES for t in (1, 2)]
    with ThreadPoolExecutor(len(jobs)) as pool:
        res = dict(zip(jobs, pool.map(lambda j: run(*j), jobs)))
    rows = {}
    for n in CASES:
        t1, t2 = res[(n, 1)], res[(n, 2)]
        rows[n] = {"tier1": t1, "tier2": t2, "rel_err": {
            "ttft_p50": t1["ttft_ms"]["p50"] / t2["ttft_ms"]["p50"] - 1,
            "ttft_p99": t1["ttft_ms"]["p99"] / t2["ttft_ms"]["p99"] - 1,
            "tpot_p50": t1["tpot_ms"]["p50"] / t2["tpot_ms"]["p50"] - 1,
            "tpot_p99": t1["tpot_ms"]["p99"] / t2["tpot_ms"]["p99"] - 1,
            "tokens_per_s_gpu": t1["tokens_per_s_gpu"] / t2["tokens_per_s_gpu"] - 1}}
        print(n, "Tier-1 vs Tier-2", json.dumps({k: f"{v:+.1%}" for k, v in rows[n]["rel_err"].items()}),
              "| Tier-2", json.dumps({"tok/s/GPU": round(t2["tokens_per_s_gpu"], 1), "ttft": t2["ttft_ms"],
                                      "tpot": t2["tpot_ms"], "queue": t2["queue"]}), flush=True)
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0052", "cost_model_version": cmv, "source_identity": source_identity(),
         "cases": {n: {"deployment": d.__dict__, "workload": w} for n, (d, w) in CASES.items()},
         "rows": rows}, indent=1))
