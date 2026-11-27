"""E0007: Phase 1 reference baseline (final Phase 1 cost model).

Frozen realistic-10-seed42 workload, Kimi-K3 official precision, P/D with one
16-GPU instance per side. Each layout runs under 'peak' (roofline upper
bound) and its calibrated/transferred profile (validation/cost_profiles.json).

    ./ie.sh experiments/E0007_phase1_baseline/run.py
"""
import json, subprocess, tempfile, time
from pathlib import Path
from adapters.cost_model import cost_model_version
from adapters.engines import ROOT, SIMTHESIZER, source_identity

PROFILES = json.loads((ROOT / "validation" / "cost_profiles.json").read_text())
CASES = [("h200", 4, "h200_dense_v1"), ("b300", 4, "b300_transfer_v1"),
         ("b300", 2, "b300_transfer_v1")]
FLAG = {"compute": "--compute-efficiency", "memory": "--memory-efficiency",
        "network": "--network-efficiency", "kernel_overhead_us": "--kernel-overhead-us"}

rows = []
with tempfile.TemporaryDirectory(prefix="e0007-") as tmp:
    for hw, tp, calibrated in CASES:
        for profile in ("peak", calibrated):
            out = Path(tmp) / f"{hw}-{tp}-{profile}"
            cmd = ["bash", "run_kimi_k3_shared_workload.sh", "--output-dir", str(out),
                   "--gpu-model", hw, "--prefill-gpu-model", hw, "--decode-gpu-model", hw,
                   "--tp", str(tp), "--prefill-tp", str(tp), "--decode-tp", str(tp)]
            for k, v in PROFILES[profile]["efficiency"].items():
                cmd += [FLAG[k], str(v)]
            t0 = time.perf_counter()
            subprocess.run(cmd, cwd=SIMTHESIZER, check=True, capture_output=True)
            m = json.loads((out / "metrics.json").read_text())
            row = {"hardware": hw, "tp": tp, "profile": profile, "gpus": 32,
                   "ttft_ms": {q: m["ttft"][q] * 1e3 for q in ("mean", "p50", "p99")},
                   "tpot_ms": {q: m["tpot"][q] * 1e3 for q in ("mean", "p50", "p99")},
                   "tokens_per_s_gpu": m["tokens_per_gpu"]["mean"],
                   "tokens_per_s_user": m["tokens_per_user"]["mean"],
                   "makespan_s": m["makespan"]["mean"], "host_wall_s": time.perf_counter() - t0}
            rows.append(row)
            print(f"{hw} tp{tp} {profile:18s} TTFT {row['ttft_ms']['mean']:7.2f} ms  TPOT "
                  f"{row['tpot_ms']['mean']:6.2f} ms  {row['tokens_per_s_gpu']:6.2f} tok/s/GPU  "
                  f"{row['tokens_per_s_user']:6.1f} tok/s/user  wall {row['host_wall_s']:.0f}s", flush=True)
cmv = cost_model_version()
Path(__file__).with_name(f"result-{cmv[:12]}.json").write_text(json.dumps(
    {"experiment": "E0007", "cost_model_version": cmv, "source_identity": source_identity(),
     "workload": "examples/workloads/realistic-10-seed42.jsonl", "profiles": PROFILES,
     "rows": rows}, indent=1))
