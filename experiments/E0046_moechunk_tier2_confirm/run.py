"""E0046: Tier-2 confirmation of the M-MOECHUNK optimum (128 x H200, 8K/1K).

Deployment P3 PP4xEP4 b2048 mc8 + D5 EP16 batch 64 at 11.0 req/s (the
recommended point), 1200 requests (~110 s of arrivals): Tier-2 vs Tier-1
on the same workload. Also the k = 1 deployment for the mechanism's effect
at Tier-2. Tier-2 result.jsonl kept under tier2/.

    ./ie.sh experiments/E0046_moechunk_tier2_confirm/run.py
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
WL = (1200, 11.0, 8192, 1024, 1)
BEST = Deployment(hardware="h200", prefill_instances=3, decode_instances=5, prefill_ep=4, prefill_pp=4,
                  prefill_budget=2048, decode_budget=64, decode_max_batch=64, prefill_moe_chunks=8)
CASES = {"mc8": BEST, "mc1": Deployment(**{**BEST.__dict__, "prefill_moe_chunks": 1})}


def run(name, dep, tier):
    out = HERE / "tier2" / f"{name}.jsonl"
    with tempfile.TemporaryDirectory(prefix="e0046-") as tmp:
        tmp = Path(tmp)
        cfg = tier2_config(dep) if tier == 2 else te.tier1_config(tier2_config(dep), models_for(dep))
        te.write_config(cfg, tmp / "c.toml")
        wl = workloads.write(workloads.fixed(*WL[:4], seed=WL[4]), tmp / "w.jsonl")
        res = tmp / "r.jsonl"
        if tier == 2 and out.exists():
            res = out
        else:
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(res)], check=True, capture_output=True)
            if tier == 2:
                out.parent.mkdir(exist_ok=True)
                out.write_text(res.read_text())
        m = steady.metrics(res, dep.gpus, SLO, sides=(dep.prefill_instances * dep.width("prefill"),
                                                      dep.decode_instances * dep.width("decode")))
    keep = ("tokens_per_s_gpu", "prefill_tokens_per_s_gpu", "decode_tokens_per_s_gpu", "slo_attainment", "queue")
    return {**{k: m[k] for k in keep}, "ttft_ms": m["ttft_ms"], "tpot_ms": m["tpot_ms"]}


if __name__ == "__main__":
    jobs = [(n, d, t) for n, d in CASES.items() for t in (1, 2)]
    with ThreadPoolExecutor(len(jobs)) as pool:
        res = dict(zip([(n, t) for n, _, t in jobs], pool.map(lambda j: run(*j), jobs)))
    rows = {}
    for n in CASES:
        t1, t2 = res[(n, 1)], res[(n, 2)]
        rows[n] = {"tier1": t1, "tier2": t2, "rel_err": {
            "ttft_p50": t1["ttft_ms"]["p50"] / t2["ttft_ms"]["p50"] - 1,
            "ttft_p99": t1["ttft_ms"]["p99"] / t2["ttft_ms"]["p99"] - 1,
            "tpot_p50": t1["tpot_ms"]["p50"] / t2["tpot_ms"]["p50"] - 1,
            "tokens_per_s_gpu": t1["tokens_per_s_gpu"] / t2["tokens_per_s_gpu"] - 1}}
        print(n, json.dumps({k: round(v, 4) for k, v in rows[n]["rel_err"].items()}),
              "tier2", json.dumps({"tok/s/GPU": round(t2["tokens_per_s_gpu"], 1), "ttft": t2["ttft_ms"],
                                   "tpot": t2["tpot_ms"], "queue": t2["queue"]}), flush=True)
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0046", "workload": WL, "cost_model_version": cmv, "source_identity": source_identity(),
         "cases": {n: d.__dict__ for n, d in CASES.items()}, "rows": rows}, indent=1))
