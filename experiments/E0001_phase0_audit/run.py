"""E0001: Phase 0 static audit — Kimi-K3 conformance, H200/B300 preset audit,
resident-memory feasibility across 8-256 GPUs. No timing is simulated."""
import json, sys
from pathlib import Path
from validation import capacity, hardware, kimi_conformance
from adapters.engines import source_identity
from adapters.cost_model import cost_model_version

out = Path(__file__).parent
rows = [r.row() for r in capacity.sweep()]
b300_usable = [r.row() for r in capacity.sweep(hardware=("b300",), capacity_gb=268.0)]
record = {
    "experiment": "E0001_phase0_audit",
    "source_identity": source_identity(),
    "cost_model_version": cost_model_version(),
    "assumptions": {"hbm_utilization": 0.9, "reserve_gb": 0.0,
                    "layout": "EP=N whole experts, attn TP x DP = N, PP=CP=DCP=1",
                    "kda_state_dtype": "bf16 (bridge)", "mla_cache_dtype": "fp8 (bridge)"},
    "kimi_structure_findings": kimi_conformance.structure_findings(),
    "kimi_precision": kimi_conformance.precision_findings(),
    "preset_audit": {h: hardware.audit_preset(h) for h in ("h200", "b300")},
    "capacity": rows,
    "capacity_b300_268gb_sensitivity": b300_usable,
}
tag = record["cost_model_version"][:12]
(out / f"result-{tag}.json").write_text(json.dumps(record, indent=1, default=str))
lines = ["| hw | GPUs | aTP | status | weights GB/GPU | free GB/GPU | req@8k | req@32k | req@128k |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    m = r["max_requests"] or {}
    lines.append(f"| {r['hardware']} | {r['n_gpus']} | {r['attn_tp']} | {r['status']} | "
                 f"{r['weight_bytes_per_gpu']/1e9:.1f} | {r['free_bytes_per_gpu']/1e9:.1f} | "
                 f"{m.get(8192,'-')} | {m.get(32768,'-')} | {m.get(131072,'-')} |")
(out / f"capacity-{tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(json.dumps(record["preset_audit"], indent=1))
