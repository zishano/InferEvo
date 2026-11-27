"""E0033: PP throughput with overlapping batches (simthesizer f462f7c), Tier-2.

Burst of 8 requests, ISL 2048 / OSL 2, prefill budget 2048 (one request per
round), P1:D1, decode PP1xEP16. Prefill: PP1xEP16 (16 GPUs), PP2xEP16
(32 GPUs, same stage layout), PP2xEP8 (16 GPUs, MoE inside one node).
Measures first-round latency and the steady completion interval.

    ./ie.sh experiments/E0033_pp_overlap/run.py
"""
import json
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters import tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import Deployment, check, tier2_config

here = Path(__file__).parent
N, ISL, OSL = 8, 2048, 2
SPACE = [Deployment(prefill_ep=ep, prefill_pp=pp, prefill_budget=2048) for ep, pp in ((16, 1), (16, 2), (8, 2))]


def job(dep):
    out = here / "tier2" / f"{dep.key}.jsonl"
    if not out.exists():
        with tempfile.TemporaryDirectory(prefix="e0033-") as tmp:
            tmp = Path(tmp)
            te.write_config(tier2_config(dep), tmp / "c.toml")
            wl = workloads.write(workloads.fixed(N, 1000.0, ISL, OSL, seed=33), tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(tmp / "r.jsonl")],
                           check=True, capture_output=True)
            out.parent.mkdir(exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    ttft = sorted(r["sub_requests"][0]["token_latency"][0] * 1e3 for r in rows)
    gaps = [b - a for a, b in zip(ttft, ttft[1:])]
    return {"key": dep.key, "prefill_gpus": dep.width("prefill"), "first_ms": ttft[0],
            "interval_ms": statistics.median(gaps), "last_ms": ttft[-1], "ttft_ms": ttft}


if __name__ == "__main__":
    for d in SPACE:
        assert not check(d, ISL + OSL), check(d, ISL + OSL)
        tier2_config(d)
    with ThreadPoolExecutor(len(SPACE)) as pool:
        rows = list(pool.map(job, SPACE))
    print(f"burst {N} x ISL {ISL}, budget 2048 (1 request/round); prefill layouts, Tier-2")
    for r in rows:
        print(f"  {r['key']:40s} {r['prefill_gpus']:3d} GPUs  first {r['first_ms']:6.0f} ms  "
              f"interval {r['interval_ms']:6.0f} ms  last {r['last_ms']:6.0f} ms  "
              f"prefill tok/s/GPU {ISL / r['interval_ms'] * 1e3 / r['prefill_gpus']:6.0f}")
    (here / "result.json").write_text(json.dumps(
        {"experiment": "E0033", "rows": rows, "source_identity": source_identity(),
         "cost_model_version": cost_model_version()}, indent=1))
