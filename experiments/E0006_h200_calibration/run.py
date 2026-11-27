"""E0006: fit H200 BF16 efficiency/overhead per kernel class to Frontier data.

    ./ie.sh experiments/E0006_h200_calibration/run.py
"""
import json, subprocess
from dataclasses import asdict
from pathlib import Path
from validation import calibration as c

frontier = c.REFERENCES / "frontier"
commit = subprocess.run(["git", "-C", str(frontier), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
fits = [c.fit("dense_gemm_bf16", c.dense_gemm_samples()),
        c.fit("moe_grouped_gemm_bf16", c.moe_grouped_gemm_samples())]
for f in fits:
    print(f"{f.kernel_class:24s} n={f.n:4d} alpha={f.alpha_s*1e6:5.2f}us eta_c={f.eta_compute:.3f} "
          f"eta_m={f.eta_memory:.3f}  |rel err| median={f.median_abs_rel_err:.3f} p90={f.p90_abs_rel_err:.3f}")
Path(__file__).with_name("result.json").write_text(json.dumps({
    "experiment": "E0006", "source": "Frontier H200 Qwen3-235B-A22B kernel_only tables",
    "frontier_commit": commit, "peak_bf16": c.H200_BF16_PEAK, "hbm_bw": c.H200_HBM_BW,
    "fits": [asdict(f) for f in fits]}, indent=1))
