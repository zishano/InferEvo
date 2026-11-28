"""E0048: Tier-1 prefill surrogates under M-RANKDISPATCH + M-MOECHUNK (step D
of M-RANKDISPATCH's A-E: co-search with the existing space).

Method as E0040/E0045 (adapters/fit_layout.py, D23 gate); prefill lowering
{dispatch = ranks, moe_chunks = 8} (E0047 best), decode default. Name:
<hw>-tp<T>-ep<E>-pp<P>-mc8-rd.

    ./ie.sh experiments/E0048_rankdispatch_surrogates/run.py h200-tp4-ep4-pp4-mc8-rd
"""
import json, sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit

here = Path(__file__).parent
LAYOUTS = ["h200-tp4-ep4-pp4-mc8-rd", "h200-tp4-ep8-pp2-mc8-rd"]
OPTIONS = {"moe_chunks": 8, "dispatch": "ranks"}

if __name__ == "__main__":
    name = sys.argv[1]
    hw, tp, ep, pp = name.split("-")[:4]
    tp, ep, pp = (int(x[2:]) for x in (tp, ep, pp))
    out = here / name
    out.mkdir(exist_ok=True)
    result = fit(out, hw, tp, ep, pp, threads=24, options=OPTIONS, log=lambda *a: print(*a, flush=True))
    (out / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity(),
                                                     lowering={"kind": "grouped", **OPTIONS}), indent=1))
