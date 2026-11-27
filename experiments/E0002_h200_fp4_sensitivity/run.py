"""E0002: sensitivity of Kimi-K3 cost to the H200 MXFP4 execution path.

Preset prices FP4 natively at 1979 TFLOP/s (= FP8). Hopper has no FP4 MMA;
alternatives: W4A8 (dequant to FP8, 1979) or W4A16 (dequant to BF16, 989.5).
Metric: serial single-device roofline ratio (valid as a ratio only).
Dequantization FLOPs/bytes are NOT added, so W4A16 ratios are lower bounds.
"""
import json
from pathlib import Path
from adapters import cost_model as c
from adapters.engines import source_identity

cases = [("decode", b, 1, ctx) for b in (1, 64, 512) for ctx in (8192, 131072)] + \
        [("prefill", b, s, None) for b in (1, 4) for s in (2048, 8192)]
rows = []
for stage, b, t, ctx in cases:
    base = c.roofline_breakdown("h200", b, t, ctx)
    w4a16 = c.roofline_breakdown("h200", b, t, ctx, fp4_rate=989.5)
    rows.append({"stage": stage, "batch": b, "tokens": t, "context": ctx,
                 "preset_us": base["serial_roofline_us"],
                 "w4a16_us": w4a16["serial_roofline_us"],
                 "ratio": w4a16["serial_roofline_us"] / base["serial_roofline_us"],
                 "fp4_compute_share_preset": base["compute_us_by_dtype"].get("fp4", 0) / base["serial_roofline_us"]})
out = Path(__file__).parent
rec = {"experiment": "E0002_h200_fp4_sensitivity", "source_identity": source_identity(),
       "cost_model_version": c.cost_model_version(), "rows": rows}
(out / f"result-{rec.get('cost_model_version', '')[:12] or 'unversioned'}.json").write_text(json.dumps(rec, indent=1))
for r in rows:
    print(f"{r['stage']:7s} b={r['batch']:<4d} t={r['tokens']:<5d} ctx={r['context']}  ratio={r['ratio']:.3f}")
