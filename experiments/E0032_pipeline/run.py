"""E0032: pipeline parallelism (PP) vs PP1 at equal GPUs per instance, Tier-2.

One request, ISL 8192 / OSL 64, P1:D1 with the same layout on both sides,
attention TP4. 16-GPU instances: PP1xEP16, PP2xEP8, PP4xEP4; 32-GPU: PP1xEP32,
PP2xEP16, PP4xEP8. Roofline peak. Single request: rounds do not overlap,
so this measures latency (MoE all-to-all span vs stage-boundary sends),
not PP throughput.

    ./ie.sh experiments/E0032_pipeline/run.py
"""
import json
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters import tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import Deployment, check, tier2_config

here = Path(__file__).parent
ISL, OSL = 8192, 64
LAYOUTS = [(16, 1), (8, 2), (4, 4), (32, 1), (16, 2), (8, 4)]       # (EP per stage, PP)
SPACE = [Deployment(prefill_ep=ep, decode_ep=ep, prefill_pp=pp, decode_pp=pp, prefill_budget=2048)
         for ep, pp in LAYOUTS]


def job(dep):
    out = here / "tier2" / f"{dep.key}.jsonl"
    if not out.exists():
        with tempfile.TemporaryDirectory(prefix="e0032-") as tmp:
            tmp = Path(tmp)
            te.write_config(tier2_config(dep), tmp / "c.toml")
            wl = workloads.write(workloads.fixed(1, 1.0, ISL, OSL, seed=31), tmp / "w.jsonl")
            t0 = time.perf_counter()
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(tmp / "r.jsonl")],
                           check=True, capture_output=True)
            wall = time.perf_counter() - t0
            out.parent.mkdir(exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
            (out.with_suffix(".wall")).write_text(f"{wall:.1f}\n")
    lat = json.loads(out.read_text().splitlines()[0])["sub_requests"][0]["token_latency"]
    return {"key": dep.key, "gpus_per_instance": dep.width("prefill"), "ep": dep.prefill_ep, "pp": dep.prefill_pp,
            "ttft_ms": lat[0] * 1e3, "tpot_ms": statistics.fmean(lat[1:]) * 1e3,
            "wall_s": float(out.with_suffix(".wall").read_text())}


if __name__ == "__main__":
    for d in SPACE:
        assert not check(d, ISL + OSL), (d.key, check(d, ISL + OSL))
        tier2_config(d)                                    # launcher configs, serially
    with ThreadPoolExecutor(len(SPACE)) as pool:
        rows = list(pool.map(job, SPACE))
    base = {r["gpus_per_instance"]: r for r in rows if r["pp"] == 1}
    print(f"1 request ISL {ISL} OSL {OSL}, P1:D1, TP4, Tier-2 (roofline peak)")
    for r in rows:
        b = base[r["gpus_per_instance"]]
        print(f"  {r['gpus_per_instance']:3d} GPUs/inst PP{r['pp']} x EP{r['ep']:<3d}  TTFT {r['ttft_ms']:8.1f} ms "
              f"({r['ttft_ms'] / b['ttft_ms'] - 1:+.1%})  TPOT {r['tpot_ms']:6.2f} ms "
              f"({r['tpot_ms'] / b['tpot_ms'] - 1:+.1%})  wall {r['wall_s']:.0f} s")
    (here / "result.json").write_text(json.dumps(
        {"experiment": "E0032", "isl": ISL, "osl": OSL, "rows": rows,
         "source_identity": source_identity(), "cost_model_version": cost_model_version()}, indent=1))
