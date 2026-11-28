"""E0062: decode surrogates for larger decode instances (EP32 / EP64, TP4).

Graph bounds (no contention) say larger decode instances raise decode
tokens/s/GPU and cut TPOT: doubling EP halves each GPU's expert bytes while
its share of the batch stays (H200 EP16 b128 281 -> EP32 b256 387 -> EP64
b512 493 tok/s/GPU). The search fixed instances at 16 GPUs, so this was never
tried. Fits per E0056 (decode-only, validated batch) with tiny-prompt runs
so decode rounds reach the target batch:
  fit run C:      B calls x (16 in, 96 out), max batch B
  held-out run D: 1.5 B calls x (32 in, 128 out)
max_requests = largest held-out round if its TPOT error passes D23, else
half. Prefill = the layout's own surrogate is not needed (decode-only use):
the fit uses the E0040 EP16 prefill model for the loaded runs' prefill side.
Name: <hw>-tp4-ep<E>-pp1-b<B>.

    ./ie.sh experiments/E0062_large_decode_instances/run.py <name>
"""
import json
import sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit, layout_dep
from validation.decode_rounds import reconstruct_multi

HERE = Path(__file__).parent
E = HERE.parent
FITS = {f"{hw}-tp4-ep{ep}-pp1-b{b}": (hw, ep, b) for hw in ("h200", "b300") for ep, b in ((32, 256), (64, 512))}


def main(name):
    hw, ep, batch = FITS[name]
    out = HERE / name
    out.mkdir(exist_ok=True)
    runs = ((batch, 1000.0, 16, 96, 62), (batch * 3 // 2, 1000.0, 32, 128, 63))
    prefill = json.loads((E / f"E0040_grouped_surrogates/{hw}-tp4-ep16-pp1/result.json").read_text())["prefill"]
    prefill["key"] = dict(prefill["key"], ep=ep, groups=ep // 4)      # decode-only fit: prefill side unused
    result = fit(out, hw, 4, ep, 1, decode_requests=batch, prefill=prefill, runs=runs, threads=24,
                 log=lambda *a: print(*a, flush=True))
    dep = layout_dep(hw, 4, ep, 1, decode_batch=batch)
    sizes = {r: max(len(x.requests) for rs in reconstruct_multi(out / "tier2" / f"{r}-{dep.key}.jsonl").values()
                    for x in rs) for r in ("A", "B")}
    ok = abs(result["runs"]["B"]["rel_err_calibrated"]["tpot_p50"]) <= 0.05
    result["decode"]["key"]["max_requests"] = min(batch, sizes["B"]) if ok else batch // 2
    result.update({"max_round_requests": {"C(fit)": sizes["A"], "D(held-out)": sizes["B"]},
                   "validated_max_requests": result["decode"]["key"]["max_requests"],
                   "source_identity": source_identity(), "lowering": "grouped"})
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(name, "round sizes", sizes, "validated_max_requests", result["validated_max_requests"],
          "held-out TPOT", f"{result['runs']['B']['rel_err_calibrated']['tpot_p50']:+.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
