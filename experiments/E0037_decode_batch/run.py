"""E0037: decode surrogate for max batch up to 128, H200 EP16 attention TP4.

Prefill keeps the E0027 TP4 model (not recalibrated). Decode: graph bounds of
batches <= 128 requests (KV to 48k), one load scale fitted on run A (64
requests, ISL 512, OSL 32, 40 req/s: decode batches up to ~64) and checked on
held-out run B (96 requests, ISL 1024, OSL 24, 30 req/s). Batches above ~96
rest on bounds + the scale (no Tier-2 check).

    ./ie.sh experiments/E0037_decode_batch/run.py
"""
import json
from pathlib import Path

from adapters.engines import source_identity
from adapters.fit_layout import fit

here = Path(__file__).parent

if __name__ == "__main__":
    E = here.parent
    prefill = json.loads((E / "E0027_long_context_coverage/result.json").read_text())["tp4"]["prefill"]
    result = fit(here, "h200", 4, 16, 1, decode_requests=128, prefill=prefill,
                 runs=((64, 40.0, 512, 32, 37), (96, 30.0, 1024, 24, 38)), log=lambda *a: print(*a, flush=True))
    (here / "result.json").write_text(json.dumps(dict(result, source_identity=source_identity()), indent=1))
