"""E0050: Tier-1 surrogates under M-FP8LINEAR (linear = fp8): decode H200
EP16 TP4 and prefill PP4xEP4 (step D of its A-E: co-search at capacity level).

E0041 method: prefill = E0040 h200-tp4-ep16-pp1 model (unchanged); decode =
graph bounds of FP8 templates (<= 64 requests, KV to 48k), one load scale
fitted on run C (64 simultaneous, ISL 64, OSL 48: rounds of ~64), held-out
run D (96 simultaneous, ISL 96, OSL 32; max batch 64). `max_requests` = the
largest round in D if D's TPOT error passes D23, else 32. Name: <layout>-fp8.
Quality of FP8 numerics is not simulated (spec M-FP8LINEAR [quality]).

    ./ie.sh experiments/E0050_fp8_surrogates/run.py
"""
import json
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit, layout_dep
from validation.decode_rounds import reconstruct_multi

HERE = Path(__file__).parent
E = HERE.parent
# Held-out D with short prompts: with 128-token prompts the faster FP8 decode
# drained calls while the default prefill fed them (rounds peaked at 48 < 64).
RUN_C, RUN_D = (64, 1000.0, 64, 48, 39), (96, 1000.0, 96, 32, 40)
DECODE = {"linear": "fp8"}

PREFILL = {"moe_chunks": 8, "dispatch": "ranks", "linear": "fp8"}     # fastest prefill (E0047) + FP8


def prefill_fit():
    """Prefill side: PP4xEP4 with PREFILL options, E0048 method (name -mc8-rd-fp8)."""
    out = HERE / "h200-tp4-ep4-pp4-mc8-rd-fp8"
    out.mkdir(exist_ok=True)
    result = fit(out, "h200", 4, 4, 4, threads=24, options=PREFILL, log=lambda *a: print(*a, flush=True))
    (out / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity(),
                                                     quality="UNVERIFIED (M-FP8LINEAR)",
                                                     lowering={"kind": "grouped", **PREFILL}), indent=1))


if __name__ == "__main__" and "prefill" in __import__("sys").argv:
    prefill_fit()
elif __name__ == "__main__":
    out = HERE / "h200-tp4-ep16-pp1-fp8"
    out.mkdir(exist_ok=True)
    prefill = json.loads((E / "E0040_grouped_surrogates/h200-tp4-ep16-pp1/result.json").read_text())["prefill"]
    result = fit(out, "h200", 4, 16, 1, decode_requests=64, prefill=prefill, runs=(RUN_C, RUN_D),
                 decode_options=DECODE, log=lambda *a: print(*a, flush=True))
    dep = layout_dep("h200", 4, 16, 1, decode_batch=64, decode_options=DECODE)
    sizes = {r: max(len(x.requests) for rs in reconstruct_multi(out / "tier2" / f"{r}-{dep.key}.jsonl").values()
                    for x in rs) for r in ("A", "B")}
    ok = abs(result["runs"]["B"]["rel_err_calibrated"]["tpot_p50"]) <= 0.05
    result["decode"]["key"]["max_requests"] = min(64, sizes["B"]) if ok else 32
    result.update({"max_round_requests": {"C(fit)": sizes["A"], "D(held-out)": sizes["B"]},
                   "validated_max_requests": result["decode"]["key"]["max_requests"],
                   "quality": "UNVERIFIED (M-FP8LINEAR)", "source_identity": source_identity(),
                   "lowering": {"kind": "grouped", **DECODE}})
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("round sizes", sizes, "validated_max_requests", result["validated_max_requests"])
