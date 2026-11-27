"""E0040: Tier-1 surrogates re-fitted under the grouped Tier-2 lowering (MC-5).

Every Tier-1 layout used by the controller, fitted and load-calibrated with
adapters/fit_layout.py (prefill Tier-2 samples, decode graph bounds, scale per
side on run A, held-out run B; D23 gate). Name: <hw>-tp<T>-ep<E>-pp<P>.

    ./ie.sh experiments/E0040_grouped_surrogates/run.py h200-tp4-ep16-pp1
"""
import json, sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit

here = Path(__file__).parent
LAYOUTS = ["h200-tp4-ep16-pp1", "h200-tp8-ep16-pp1", "h200-tp4-ep8-pp2", "h200-tp4-ep4-pp4",
           "h200-tp8-ep8-pp2", "b300-tp4-ep16-pp1"]

if __name__ == "__main__":
    name = sys.argv[1]
    hw, *rest = name.split("-")
    tp, ep, pp = (int(part[2:]) for part in rest)
    out = here / name
    out.mkdir(exist_ok=True)
    result = fit(out, hw, tp, ep, pp, threads=24, log=lambda *a: print(*a, flush=True))
    (out / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity(),
                                                     lowering="grouped"), indent=1))
