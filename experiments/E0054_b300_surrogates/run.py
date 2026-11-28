"""E0054: B300 Tier-1 surrogates for the layouts and mechanisms the H200 optimum
uses (Phase 6: B300 search). Methods: E0040 (layouts), E0045/E0048 (prefill
mechanism options), E0041 (decode batch 64), E0050 (FP8, quality UNVERIFIED).
Names follow agent.controller (<hw>-tp<T>-ep<E>-pp<P><option suffix>);
`b300-tp4-ep16-pp1-b64` is the decode batch-64 validation (LARGE_DECODE).

    ./ie.sh experiments/E0054_b300_surrogates/run.py <name>
"""
import json
import sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit, layout_dep
from validation.decode_rounds import reconstruct_multi

HERE = Path(__file__).parent
E = HERE.parent
RUN_C, RUN_D = (64, 1000.0, 64, 48, 39), (96, 1000.0, 96, 32, 40)    # E0041 / E0050 batch-64 validation
RD = {"moe_chunks": 8, "dispatch": "ranks"}
FITS = {  # name: (tp, ep, pp, prefill options, decode options, decode batch-64 validation)
    "b300-tp4-ep4-pp4": (4, 4, 4, {}, {}, False),
    "b300-tp4-ep8-pp2": (4, 8, 2, {}, {}, False),
    "b300-tp4-ep4-pp4-mc8": (4, 4, 4, {"moe_chunks": 8}, {}, False),
    "b300-tp4-ep4-pp4-mc8-rd": (4, 4, 4, RD, {}, False),
    "b300-tp4-ep8-pp2-mc8-rd": (4, 8, 2, RD, {}, False),
    "b300-tp4-ep4-pp4-mc8-rd-fp8": (4, 4, 4, {**RD, "linear": "fp8"}, {}, False),
    "b300-tp4-ep16-pp1-b64": (4, 16, 1, {}, {}, True),
    "b300-tp4-ep16-pp1-fp8": (4, 16, 1, {}, {"linear": "fp8"}, True),
}


def main(name):
    tp, ep, pp, opts, dopts, batch64 = FITS[name]
    out = HERE / name
    out.mkdir(exist_ok=True)
    log = lambda *a: print(*a, flush=True)
    if not batch64:
        result = fit(out, "b300", tp, ep, pp, threads=24, options=opts, decode_options=dopts, log=log)
    else:
        prefill = json.loads((E / "E0040_grouped_surrogates/b300-tp4-ep16-pp1/result.json").read_text())["prefill"]
        result = fit(out, "b300", tp, ep, pp, decode_requests=64, prefill=prefill, runs=(RUN_C, RUN_D),
                     decode_options=dopts, log=log)
        dep = layout_dep("b300", tp, ep, pp, decode_batch=64, decode_options=dopts)
        sizes = {r: max(len(x.requests) for rs in reconstruct_multi(out / "tier2" / f"{r}-{dep.key}.jsonl").values()
                        for x in rs) for r in ("A", "B")}
        ok = abs(result["runs"]["B"]["rel_err_calibrated"]["tpot_p50"]) <= 0.05
        result["decode"]["key"]["max_requests"] = min(64, sizes["B"]) if ok else 32
        result.update({"max_round_requests": {"C(fit)": sizes["A"], "D(held-out)": sizes["B"]},
                       "validated_max_requests": result["decode"]["key"]["max_requests"]})
    quality = "UNVERIFIED (M-FP8LINEAR)" if "fp8" in name else None
    (out / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity(),
        lowering={"kind": "grouped", **opts} if opts else "grouped",
        **({"decode_lowering": {"kind": "grouped", **dopts}} if dopts else {}),
        **({"quality": quality} if quality else {})), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
