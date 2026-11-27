"""E0041: decode surrogate up to 128 requests per round, H200 EP16 TP4, grouped lowering.

Prefill: the E0040 h200-tp4-ep16-pp1 model (unchanged). Decode: graph bounds
(<= 128 requests, KV to 48k, OOM batches skipped), one load scale fitted on run
C (64 simultaneous requests, ISL 64, OSL 48: decode rounds of ~64) and checked
on held-out run D (96 simultaneous requests, ISL 128, OSL 32: rounds of ~96).
The reconstructed round sizes are recorded; `validated_max_requests` is the
largest round size seen in D if the D TPOT error passes D23, else 32.

    ./ie.sh experiments/E0041_grouped_decode_batch/run.py
"""
import json
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit, layout_dep
from validation.decode_rounds import reconstruct_multi

here = Path(__file__).parent
E = here.parent
RUN_C, RUN_D = (64, 1000.0, 64, 48, 39), (96, 1000.0, 128, 32, 40)

if __name__ == "__main__":
    prefill = json.loads((E / "E0040_grouped_surrogates/h200-tp4-ep16-pp1/result.json").read_text())["prefill"]
    result = fit(here, "h200", 4, 16, 1, decode_requests=128, prefill=prefill, runs=(RUN_C, RUN_D),
                 log=lambda *a: print(*a, flush=True))
    dep = layout_dep("h200", 4, 16, 1, decode_batch=128)
    sizes = {r: max(len(x.requests) for rs in reconstruct_multi(here / "tier2" / f"{r}-{dep.key}.jsonl").values()
                    for x in rs)
             for r in ("A", "B")}
    ok = abs(result["runs"]["B"]["rel_err_calibrated"]["tpot_p50"]) <= 0.05
    cap = min(128, sizes["B"]) if ok else 32
    result["decode"]["key"]["max_requests"] = cap
    result.update({"max_round_requests": {"C(fit)": sizes["A"], "D(held-out)": sizes["B"]},
                   "validated_max_requests": cap, "source_identity": source_identity(), "lowering": "grouped"})
    (here / "result.json").write_text(json.dumps(result, indent=1))
    print("round sizes", sizes, "validated_max_requests", cap)
