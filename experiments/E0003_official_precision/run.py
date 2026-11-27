"""E0003: effect of MC-1 (official precision) + CM-5 (MLA prefill MHA width).

Run once per cost model (INFEREVO_SIMTHESIZER selects the tree); writes
result-<label>.json. compare.py diffs two labels.
"""
import json, os, sys
from pathlib import Path
from adapters import cost_model
from validation import capacity
from adapters.engines import source_identity, SIMTHESIZER

label = sys.argv[1]
cases = [("decode", 64, 1, 8192), ("decode", 512, 1, 131072),
         ("prefill", 1, 8192, None), ("prefill", 1, 32768, None), ("prefill", 4, 8192, None)]
rows = []
for hw in ("h200", "b300"):
    for stage, b, t, ctx in cases:
        r = cost_model.roofline_breakdown(hw, b, t, ctx)
        rows.append({"hw": hw, "stage": stage, "batch": b, "tokens": t, "context": ctx,
                     "serial_roofline_us": r["serial_roofline_us"],
                     "compute_us_by_dtype": r["compute_us_by_dtype"]})
cap = [c.row() for c in capacity.sweep(gpu_counts=(8, 16, 32, 64, 128))]
rec = {"label": label, "simthesizer_tree": str(SIMTHESIZER),
       "cost_model_version": cost_model.cost_model_version(),
       "source_identity": source_identity(), "roofline": rows, "capacity": cap}
Path(__file__).with_name(f"result-{label}.json").write_text(json.dumps(rec, indent=1, default=str))
print(label, rec["cost_model_version"][:12])
