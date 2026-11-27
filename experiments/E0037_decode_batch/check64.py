"""E0037 check: held-out decode validation at batch ~64 (run C, Tier-2).

Run C: 64 requests, ISL 64, OSL 48, simultaneous arrival -> decode rounds of
~64 requests. Not used for fitting. Compares the refit decode model with
Tier-2 per reconstructed round and on TPOT p50; result.json gets
`validated_max_requests` (64 if the round median and TPOT errors are <= 5 %).

    ./ie.sh experiments/E0037_decode_batch/check64.py
"""
import json, statistics
from pathlib import Path

from adapters import tier1
from adapters.fit_layout import _loaded, layout_dep
from validation.decode_rounds import reconstruct_multi

here = Path(__file__).parent
E = here.parent
RUN_C = (64, 1000.0, 64, 48, 39)

if __name__ == "__main__":
    dep = layout_dep("h200", 4, 16, 1, decode_batch=128)
    out = json.loads((here / "result.json").read_text())
    pre = tier1.Tier1Model(**json.loads((E / "E0027_long_context_coverage/result.json").read_text())["tp4"]["prefill"])
    dec = tier1.Tier1Model(**out["decode"])
    path = here / "tier2" / f"C-{dep.key}.jsonl"
    t2 = _loaded(dep, RUN_C, path)
    t1 = _loaded(dep, RUN_C, None, {"p": pre, "d": dec})
    rounds = [([(i, 1, kv) for i, kv in zip(r.requests, r.kv_lens)], r.duration_s)
              for rs in reconstruct_multi(path).values() for r in rs if r.duration_s is not None and not r.first_decode]
    big = [(b, y) for b, y in rounds if len(b) >= 48]
    err = [(dec.step_time(b) - y) / y for b, y in big]
    tpot = t1["tpot_p50"] / t2["tpot_p50"] - 1
    ok = bool(big) and abs(statistics.median(err)) <= 0.05 and abs(tpot) <= 0.05
    out["check_C"] = {"run": RUN_C, "tier2": t2, "tpot_p50_rel_err": tpot, "rounds_ge48": len(big),
                      "max_round_requests": max((len(b) for b, _ in rounds), default=0),
                      "round_err_median": statistics.median(err) if err else None,
                      "round_err_max_abs": max(map(abs, err)) if err else None}
    out["decode"]["key"]["max_requests"] = 64 if ok else 32
    out["validated_max_requests"] = out["decode"]["key"]["max_requests"]
    (here / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["check_C"], indent=1), "validated_max_requests", out["validated_max_requests"])
