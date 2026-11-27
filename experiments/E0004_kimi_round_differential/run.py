"""E0004: Kimi shared-path round differential (validation/round_bounds.py).

Runs the engine's Kimi P/D smoke case (one request: 5 input, 3 output tokens,
routing seed 0) with reports for each (hardware, attention TP), then checks
every engine round against bounds computed independently from the Python
bridge's exported graph for the same batch.

    ./ie.sh experiments/E0004_kimi_round_differential/run.py
"""
import csv, json, subprocess, sys, tempfile
from pathlib import Path
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, enable_bridge_imports, source_identity
from validation.round_bounds import round_bounds

enable_bridge_imports()
from rooflang_bridge.kimi_k3_shared import KimiTemplate  # noqa: E402

CASES = [("h200", 4), ("h200", 8), ("b300", 2), ("b300", 4)]
# Round batches of the fixed request: P prefills 5 tokens; D decodes twice.
BATCHES = {("p0", 0): ("p", {"q_len": 5, "kv_len": 0, "lm_head_len": 1}),
           ("d0", 0): ("d", {"q_len": 1, "kv_len": 5, "lm_head_len": 1}),
           ("d0", 1): ("d", {"q_len": 1, "kv_len": 6, "lm_head_len": 1})}


def engine_rounds(hw, tp, out):
    cmd = ["bash", "run_kimi_k3_shared.sh", "--reports", "--output-dir", str(out),
           "--gpu-model", hw, "--prefill-gpu-model", hw, "--decode-gpu-model", hw,
           "--tp", str(tp), "--prefill-tp", str(tp), "--decode-tp", str(tp)]
    subprocess.run(cmd, cwd=SIMTHESIZER, check=True, capture_output=True)
    with open(out / "rounds.csv") as f:
        return {(r["instance_id"], int(r["round"])): float(r["duration_s"]) for r in csv.DictReader(f)}


rows = []
with tempfile.TemporaryDirectory(prefix="e0004-") as tmp:
    for hw, tp in CASES:
        measured = engine_rounds(hw, tp, Path(tmp) / f"{hw}-tp{tp}")
        assert set(measured) == set(BATCHES), measured
        for key, (stage, batch) in BATCHES.items():
            ops = KimiTemplate(tp, 16, {"gpu_model": hw, "gpu_count": 16}, 0,
                               stage=stage).export([batch], [0])["operations"]
            b = round_bounds(ops, hw)
            problems = b.check(measured[key])
            rows.append({"hardware": hw, "tp": tp, "instance": key[0], "round": key[1],
                         "engine_ms": measured[key] * 1e3, "lower_ms": b.lower_s * 1e3,
                         "upper_ms": b.upper_s * 1e3, "critical_path_ms": b.critical_path_s * 1e3,
                         "max_resource_ms": b.max_resource_load_s * 1e3,
                         "max_resource": b.max_resource, "operations": b.operations,
                         "engine_over_lower": measured[key] / b.lower_s,
                         "problems": problems})
            print(f"{hw} tp{tp} {key}: engine {measured[key]*1e3:.3f} ms in "
                  f"[{b.lower_s*1e3:.3f}, {b.upper_s*1e3:.3f}] ms "
                  f"({'OK' if not problems else problems[:2]})", flush=True)
cmv = cost_model_version()
Path(__file__).with_name(f"result-{cmv[:12]}.json").write_text(json.dumps(
    {"experiment": "E0004", "cost_model_version": cmv,
     "source_identity": source_identity(), "rows": rows}, indent=1))
sys.exit(1 if any(r["problems"] for r in rows) else 0)
