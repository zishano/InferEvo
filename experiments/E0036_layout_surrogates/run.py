"""E0036: Tier-1 surrogates + load calibration for more 16-GPU layouts (H200).

  tp4-ep4-pp4: PP4 x EP4, attention TP4 (4 GPUs per stage, half a node)
  tp8-ep8-pp2: PP2 x EP8, attention TP8 (one node per stage, no attention DP)
Method: adapters/fit_layout.py (E0034 fit + E0035 calibration, held-out run B).

    ./ie.sh experiments/E0036_layout_surrogates/run.py tp4-ep4-pp4
"""
import sys
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit

here = Path(__file__).parent

if __name__ == "__main__":
    name = sys.argv[1]
    tp, ep, pp = (int(part[2:]) for part in name.split("-"))
    out = here / name
    out.mkdir(exist_ok=True)
    result = fit(out, "h200", tp, ep, pp, threads=30, log=lambda *a: print(*a, flush=True))
    import json
    (out / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity()), indent=1))
