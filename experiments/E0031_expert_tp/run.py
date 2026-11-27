"""E0031: expert TP (ETP) vs whole-expert EP at equal GPU count, Tier-2.

One request, ISL 8192 / OSL 64, P1:D1 with the same layout on both sides,
attention TP4. 16-GPU instances: EP16xETP1, EP8xETP2, EP4xETP4, EP2xETP8;
32-GPU instances: EP32xETP1, EP16xETP2. Roofline peak (no derate), so this
isolates the mechanism: per-GPU expert weight reads, critical-path balance
and the extra intra-group scatter/gather + reduce of ETP.

    ./ie.sh experiments/E0031_expert_tp/run.py
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
LAYOUTS = [(16, 1), (8, 2), (4, 4), (2, 8), (32, 1), (16, 2)]
SPACE = [Deployment(prefill_ep=ep, decode_ep=ep, prefill_etp=etp, decode_etp=etp, prefill_budget=2048)
         for ep, etp in LAYOUTS]


def job(dep):
    out = here / "tier2" / f"{dep.key}.jsonl"
    if not out.exists():
        with tempfile.TemporaryDirectory(prefix="e0031-") as tmp:
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
    return {"key": dep.key, "gpus_per_instance": dep.width("prefill"), "ep": dep.prefill_ep,
            "etp": dep.prefill_etp, "ttft_ms": lat[0] * 1e3, "tpot_ms": statistics.fmean(lat[1:]) * 1e3,
            "wall_s": float(out.with_suffix(".wall").read_text())}


if __name__ == "__main__":
    for d in SPACE:
        assert not check(d, ISL + OSL), (d.key, check(d, ISL + OSL))
        tier2_config(d)                                    # launcher configs, serially
    with ThreadPoolExecutor(len(SPACE)) as pool:
        rows = list(pool.map(job, SPACE))
    base = {r["gpus_per_instance"]: r for r in rows if r["etp"] == 1}
    print(f"1 request ISL {ISL} OSL {OSL}, P1:D1, TP4, Tier-2 (roofline peak)")
    for r in rows:
        b = base[r["gpus_per_instance"]]
        print(f"  {r['gpus_per_instance']:3d} GPUs/inst EP{r['ep']:<3d} x ETP{r['etp']}  TTFT {r['ttft_ms']:8.1f} ms "
              f"({r['ttft_ms'] / b['ttft_ms'] - 1:+.1%})  TPOT {r['tpot_ms']:6.2f} ms "
              f"({r['tpot_ms'] / b['tpot_ms'] - 1:+.1%})  wall {r['wall_s']:.0f} s")
    (here / "result.json").write_text(json.dumps(
        {"experiment": "E0031", "isl": ISL, "osl": OSL, "rows": rows,
         "source_identity": source_identity(), "cost_model_version": cost_model_version()}, indent=1))
