"""E0005: counterfactual resource probes on the Kimi P/D smoke request.

Derates one resource class at a time (CM-4) and reports TTFT/TPOT
elasticity = relative latency increase / relative rate decrease. A class
with elasticity ~1 bounds the metric; ~0 means its cost is hidden. This is a
probe of the *model*, not hardware evidence.

    ./ie.sh experiments/E0005_efficiency_sensitivity/run.py [smoke|workload]

smoke: one request (5/3 tokens). workload: frozen realistic-10-seed42 sample.
"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity

CASES = [("h200", 4), ("b300", 4), ("b300", 2)]
FACTOR = 0.5
PROBES = [None, "compute", "memory", "network"]


LABEL = sys.argv[1] if len(sys.argv) > 1 else "smoke"
SCRIPT = {"smoke": "run_kimi_k3_shared.sh", "workload": "run_kimi_k3_shared_workload.sh"}[LABEL]


def run(hw, tp, probe, out):
    cmd = ["bash", SCRIPT, "--output-dir", str(out),
           "--gpu-model", hw, "--prefill-gpu-model", hw, "--decode-gpu-model", hw,
           "--tp", str(tp), "--prefill-tp", str(tp), "--decode-tp", str(tp)]
    if probe:
        cmd += [f"--{probe}-efficiency", str(FACTOR)]
    started = time.perf_counter()
    subprocess.run(cmd, cwd=SIMTHESIZER, check=True, capture_output=True)
    m = json.loads((out / "metrics.json").read_text())
    out_m = {k: m[k]["mean"] for k in ("ttft", "tpot")}
    out_m["tokens_per_gpu"] = m["tokens_per_gpu"]["mean"]
    WALL.append({"hardware": hw, "tp": tp, "probe": probe,
                 "wall_s": time.perf_counter() - started})
    return out_m


WALL = []


rows = []
with tempfile.TemporaryDirectory(prefix="e0005-") as tmp:
    for hw, tp in CASES:
        base = None
        for probe in PROBES:
            m = run(hw, tp, probe, Path(tmp) / f"{hw}-{tp}-{probe}")
            if probe is None:
                base = m
                continue
            # rate x FACTOR -> fully bound time x 1/FACTOR
            el = {k: (m[k] / base[k] - 1) / (1 / FACTOR - 1) for k in ("ttft", "tpot")}
            rows.append({"hardware": hw, "tp": tp, "probe": probe, "factor": FACTOR,
                         "base_ms": {k: v * 1e3 for k, v in base.items()},
                         "probed_ms": {k: v * 1e3 for k, v in m.items()},
                         "elasticity": el})
            print(f"{hw} tp{tp} {probe:8s} TTFT {base['ttft']*1e3:7.3f}->{m['ttft']*1e3:7.3f} "
                  f"(e={el['ttft']:.2f})  TPOT {base['tpot']*1e3:7.3f}->{m['tpot']*1e3:7.3f} "
                  f"(e={el['tpot']:.2f})", flush=True)
cmv = cost_model_version()
Path(__file__).with_name(f"result-{LABEL}-{cmv[:12]}.json").write_text(json.dumps(
    {"experiment": "E0005", "label": LABEL, "script": SCRIPT, "cost_model_version": cmv,
     "source_identity": source_identity(), "rows": rows, "host_wall": WALL}, indent=1))
