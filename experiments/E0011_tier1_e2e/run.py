"""E0011: end-to-end Tier-1 (step_model runner) vs Tier-2 (shared Rooflang).

Same config (E0008 config.toml: H200 TP4/EP16 P1+D1, frozen realistic-10
workload, peak); Tier-1 uses E0012 prefill (engine-trained) and E0009 decode surrogates.
Tier-2 reference: E0008 result-c623652.jsonl (shared path unchanged since).

    ./ie.sh experiments/E0011_tier1_e2e/run.py
"""
import glob, json, statistics, tempfile, time
from pathlib import Path
from adapters import tier1, tier1_engine as te
from adapters.engines import ROOT, SIMTHESIZER, source_identity

E = ROOT / "experiments"
dec = tier1.Tier1Model(**json.load(open(sorted(glob.glob(str(E / "E0009_tier1_decode/result-*.json")))[-1]))["model"])
# Prefill: engine-trained surrogate (E0012); E0010 graph-bound model is 24 % optimistic.
pre = tier1.Tier1Model(**json.load(open(sorted(glob.glob(str(E / "E0012_tier1_prefill_engine/result-*.json")))[-1]))["model"])
workload = SIMTHESIZER / "examples/workloads/realistic-10-seed42.jsonl"


def per_request(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        (s,) = r["sub_requests"]
        lat = s["token_latency"]
        out[r["request_id"]] = {"ttft": lat[0], "tpot": statistics.fmean(lat[1:]) if len(lat) > 1 else None,
                                "e2e": sum(lat)}
    return out


with tempfile.TemporaryDirectory(prefix="e0011-") as tmp:
    cfg = te.tier1_config(te.load(E / "E0008_tier1_feasibility/config.toml"), {"p": pre, "d": dec})
    te.write_config(cfg, Path(tmp) / "config.toml")
    t0 = time.perf_counter()
    te.run(Path(tmp) / "config.toml", workload, Path(tmp) / "result.jsonl")
    wall = time.perf_counter() - t0
    t1 = per_request(Path(tmp) / "result.jsonl")
t2 = per_request(E / "E0008_tier1_feasibility/result-c623652.jsonl")
rows, errs = [], {"ttft": [], "tpot": [], "e2e": []}
for rid in sorted(t2):
    row = {"request": rid}
    for k in errs:
        a, b = t1[rid][k], t2[rid][k]
        if a is None or b is None:
            continue
        row[k] = {"tier1_ms": a * 1e3, "tier2_ms": b * 1e3, "rel_err": (a - b) / b}
        errs[k].append((a - b) / b)
    rows.append(row)
summary = {k: {"median_abs": statistics.median(abs(e) for e in v), "max_abs": max(abs(e) for e in v),
               "mean_signed": statistics.fmean(v)} for k, v in errs.items()}
agg = {k: {"tier1_ms": statistics.fmean(t1[r][k] for r in t2 if t1[r][k] is not None) * 1e3,
           "tier2_ms": statistics.fmean(t2[r][k] for r in t2 if t2[r][k] is not None) * 1e3}
       for k in errs}
print(json.dumps({"summary": summary, "aggregate_mean": agg, "tier1_wall_s": wall,
                  "tier2_wall_s_reference": 194.1}, indent=1))
(Path(__file__).with_name("result.json")).write_text(json.dumps({
    "experiment": "E0011", "source_identity": source_identity(), "rows": rows, "summary": summary,
    "aggregate_mean": agg, "tier1_wall_s": wall,
    "tier2_reference": "E0008 result-c623652.jsonl (194 s wall, E0008 wall profile)",
    "models": {"prefill": pre.key, "decode": dec.key}}, indent=1))
