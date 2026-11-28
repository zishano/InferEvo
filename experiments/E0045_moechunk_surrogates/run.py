"""E0045: Tier-1 prefill surrogates under M-MOECHUNK (step D of its A-E:
co-search the new variable with the existing design space).

Same method as E0040 (adapters/fit_layout.py, D23 gate) with the prefill side
at moe_chunks = 8 (E0044 best); decode stays unchunked. Name:
<hw>-tp<T>-ep<E>-pp<P>-mc<K>.

    ./ie.sh experiments/E0045_moechunk_surrogates/run.py h200-tp4-ep4-pp4-mc8
"""
import json, sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit

here = Path(__file__).parent
LAYOUTS = ["h200-tp4-ep4-pp4-mc8", "h200-tp4-ep8-pp2-mc8"]

if __name__ == "__main__":
    name = sys.argv[1]
    hw, *rest = name.split("-")
    tp, ep, pp, mc = (int(part[2:]) for part in rest)
    out = here / name
    out.mkdir(exist_ok=True)
    result = fit(out, hw, tp, ep, pp, threads=24, options={"moe_chunks": mc}, log=lambda *a: print(*a, flush=True))
    (out / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity(),
                                                     lowering={"kind": "grouped", "moe_chunks": mc}), indent=1))
