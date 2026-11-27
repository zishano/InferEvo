"""E0037 refit: decode surrogate for max batch 128 from loaded Tier-2 rounds (E0022 method).

run.py's single load scale did not transfer (held-out B TPOT +6.3 %: B is
prefill-bound, A has decode batches to ~64). Here decode is fitted on graph
bounds (<= 128 requests, KV to 48k, OOM batches skipped) plus the decode
rounds reconstructed from Tier-2 run A (validation/decode_rounds, PP1), and
checked end to end on held-out run B. Prefill: E0027 TP4, unchanged.

    ./ie.sh experiments/E0037_decode_batch/refit.py
"""
import json, multiprocessing, statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from adapters import tier1
from adapters.cost_model import cost_model_version
from adapters.engines import source_identity
from adapters.fit_layout import _anchor, _loaded, layout_dep
from validation.decode_rounds import reconstruct_multi

here = Path(__file__).parent
E = here.parent
RUNS = {"A": (64, 40.0, 512, 32, 37), "B": (96, 30.0, 1024, 24, 38)}

if __name__ == "__main__":
    dep = layout_dep("h200", 4, 16, 1, decode_batch=128)
    pre = tier1.Tier1Model(**json.loads((E / "E0027_long_context_coverage/result.json").read_text())["tp4"]["prefill"])
    batches = tier1.sample_batches("d", 120, 360, 128, 49152, 512)
    with ProcessPoolExecutor(24, mp_context=multiprocessing.get_context("fork")) as procs:
        ys = list(procs.map(_anchor, [("h200", 4, 16, 1, b) for b in batches]))
    anchors = [(b, y) for b, y in zip(batches, ys) if y is not None]
    loaded = [([(i, 1, kv) for i, kv in zip(r.requests, r.kv_lens)], r.duration_s)
              for rs in reconstruct_multi(here / "tier2" / f"A-{dep.key}.jsonl").values()
              for r in rs if r.duration_s is not None and not r.first_decode]
    train = [a for i, a in enumerate(anchors) if i % 5] + loaded
    key = {"hardware": "h200", "tp": 4, "ep": 16, "pp": 1, "groups": 4, "efficiency": {}, "stage": "d",
           "max_requests": 128, "cost_model_version": cost_model_version(),
           "target": "loaded Tier-2 decode rounds (run A) + graph bounds, <= 128 requests, KV to 48k"}
    dec = tier1.fit_samples([b for b, _ in train], [y for _, y in train], key)
    held = [abs(dec.step_time(b) - y) / y for i, (b, y) in enumerate(anchors) if i % 5 == 0]
    fit_loaded = [abs(dec.step_time(b) - y) / y for b, y in loaded]
    runs = {}
    for r, run in RUNS.items():
        t2 = _loaded(dep, run, here / "tier2" / f"{r}-{dep.key}.jsonl")
        t1 = _loaded(dep, run, None, {"p": pre, "d": dec})
        runs[r] = {"tier2": t2, "rel_err_calibrated": {k: t1[k] / t2[k] - 1 for k in t1}}
        print(r, {k: round(v, 3) for k, v in runs[r]["rel_err_calibrated"].items()}, flush=True)
    print("loaded rounds", len(loaded), "max requests in a loaded round", max(len(b) for b, _ in loaded),
          "anchor held-out median", round(statistics.median(held), 4), "loaded fit median", round(statistics.median(fit_loaded), 4))
    out = json.loads((here / "result.json").read_text())
    out.update({"method": "loaded-round refit (refit.py); run.py scale result kept in scale_attempt",
                "scale_attempt": {"scales": out["scales"], "runs": out["runs"]},
                "decode": asdict(dec), "runs": runs, "loaded_rounds": len(loaded),
                "decode_heldout": {"median": statistics.median(held), "max": max(held)},
                "source_identity": source_identity()})
    out.pop("scales", None)
    (here / "result.json").write_text(json.dumps(out, indent=1))
