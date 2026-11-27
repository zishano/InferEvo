"""E0030: parallel-strategy optimization, 128xH200, ISL 8192 / OSL 1024.

Workload: 600 fixed-length requests, Poisson. SLO (assumption): TTFT <= 5 s,
TPOT <= 50 ms. Space: 8 EP16 instances split P:D 1:7..7:1 x attention TP
{4,8} (equal P/D) x prefill budget {128,512,2048}; decode max batch 32
(launcher default; surrogates validated to 32). Metric: steady SLO capacity
(adapters/steady.py). Tier-1 (E0027 surrogates).

    ./ie.sh experiments/E0030_isl8k_osl1k_128/run.py
"""
import json, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from adapters import steady, tier1_engine as te, workloads
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from agent.controller import models_for

here = Path(__file__).parent
SLO = {"ttft_s": 5.0, "tpot_s": 0.05}
SPACE = [Deployment(prefill_instances=p, decode_instances=8 - p, prefill_budget=b, prefill_tp=t, decode_tp=t)
         for p in range(1, 8) for t in (4, 8) for b in (128, 512, 2048)]
BASELINE = Deployment(prefill_instances=4, decode_instances=4)          # balanced, launcher defaults


def run_once(dep, rate):
    cfg = te.tier1_config(tier2_config(dep), models_for(dep))
    with tempfile.TemporaryDirectory(prefix="e0030-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        te.run(tmp / "c.toml", workloads.write(workloads.fixed(600, rate, 8192, 1024, seed=30), tmp / "w.jsonl"),
               tmp / "r.jsonl")
        return steady.metrics(tmp / "r.jsonl", dep.gpus, SLO)


def job(dep):
    return dep.key, steady.capacity(lambda r: run_once(dep, r), 0.5, 2.0, steps=7)


if __name__ == "__main__":
    for d in SPACE:
        tier2_config(d)                       # build configs serially (launcher)
    with ProcessPoolExecutor(len(SPACE)) as pool:
        res = dict(pool.map(job, SPACE))
    rank = sorted(res, key=lambda k: -res[k]["rate"])
    print(f"capacity (req/s at 128 GPUs; SLO TTFT<=5 s, TPOT<=50 ms, 90 %); baseline {BASELINE.key}")
    for k in rank:
        c, a = res[k], res[k].get("at_rate")
        print("  %-38s %6.3f req/s  %s%s" % (k, c["rate"], "" if not a else
              "%5.0f out-tok/s/GPU  TTFT p50 %5.0f p99 %5.0f ms  TPOT p50 %5.1f p99 %5.1f ms" % (
                  c["rate"] * 1024 / 128, a["ttft_ms"]["p50"], a["ttft_ms"]["p99"], a["tpot_ms"]["p50"], a["tpot_ms"]["p99"]),
              "  <- baseline" if k == BASELINE.key else ""))
    (here / "result.json").write_text(json.dumps({"experiment": "E0030", "slo": SLO, "baseline": BASELINE.key,
        "ranking": rank, "capacity": res, "source_identity": source_identity()}, indent=1))
