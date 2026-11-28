"""E0056: decode surrogates validated to 128 calls per round (EP16 TP4,
H200 / B300, linear bf16 / fp8). Graph bound: batch 128 vs 64 makes a decode
round ~23 % longer for 2x the tokens (expert reads amortize), +62 % decode
throughput per instance at TPOT ~28 ms (H200).

E0041 method with workloads that actually reach 128-call rounds: earlier
held-out runs (E0041, E0050) were fed by a slow prefill and drained before
the target batch. Prompts are tiny so all calls enter decode at once:
  fit run C:      128 calls x (16 in, 96 out)
  held-out run D: 192 calls x (32 in, 128 out), max batch 128
Prefill = the hardware's E0040 EP16 model (not refitted). max_requests = the
largest round seen in D if D's TPOT error passes D23 (else 64). Name:
<hw>-tp4-ep16-pp1[-fp8]-b128. FP8 rows: quality UNVERIFIED (M-FP8LINEAR).

    ./ie.sh experiments/E0056_decode_batch128/run.py <name>
"""
import json
import sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit, layout_dep
from validation.decode_rounds import reconstruct_multi

HERE = Path(__file__).parent
E = HERE.parent
RUN_C, RUN_D = (128, 1000.0, 16, 96, 56), (192, 1000.0, 32, 128, 57)
FITS = {"h200-tp4-ep16-pp1-b128": ("h200", {}), "h200-tp4-ep16-pp1-fp8-b128": ("h200", {"linear": "fp8"}),
        "b300-tp4-ep16-pp1-b128": ("b300", {}), "b300-tp4-ep16-pp1-fp8-b128": ("b300", {"linear": "fp8"})}


def main(name):
    hw, dopts = FITS[name]
    out = HERE / name
    out.mkdir(exist_ok=True)
    prefill = json.loads((E / f"E0040_grouped_surrogates/{hw}-tp4-ep16-pp1/result.json").read_text())["prefill"]
    result = fit(out, hw, 4, 16, 1, decode_requests=128, prefill=prefill, runs=(RUN_C, RUN_D),
                 decode_options=dopts, log=lambda *a: print(*a, flush=True))
    dep = layout_dep(hw, 4, 16, 1, decode_batch=128, decode_options=dopts)
    sizes = {r: max(len(x.requests) for rs in reconstruct_multi(out / "tier2" / f"{r}-{dep.key}.jsonl").values()
                    for x in rs) for r in ("A", "B")}
    ok = abs(result["runs"]["B"]["rel_err_calibrated"]["tpot_p50"]) <= 0.05
    result["decode"]["key"]["max_requests"] = min(128, sizes["B"]) if ok else 64
    result.update({"max_round_requests": {"C(fit)": sizes["A"], "D(held-out)": sizes["B"]},
                   "validated_max_requests": result["decode"]["key"]["max_requests"],
                   "source_identity": source_identity(),
                   "decode_lowering": {"kind": "grouped", **dopts} if dopts else "grouped",
                   **({"quality": "UNVERIFIED (M-FP8LINEAR)"} if dopts else {})})
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(name, "round sizes", sizes, "validated_max_requests", result["validated_max_requests"],
          "held-out TPOT", f"{result['runs']['B']['rel_err_calibrated']['tpot_p50']:+.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
