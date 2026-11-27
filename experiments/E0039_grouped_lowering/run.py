"""E0039: grouped operator lowering (one chain per Attention group, grouped-GEMM
MoE) vs the per-request lowering, on saved Tier-2 runs.

Each reference is a kept Tier-2 result (per-request lowering) with its exact
config and workload seed; the same run is repeated with
`model_runner.lowering = "grouped"`. Reports per-run TTFT/TPOT deltas and
wall time; acceptance target: |TTFT p50|, |TPOT p50| deltas <= 3 %.

    ./ie.sh experiments/E0039_grouped_lowering/run.py
"""
import copy, json, statistics, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters import tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity
from adapters.fit_layout import RUN_A, RUN_B, layout_dep
from adapters.topologies import Deployment, tier2_config

here = Path(__file__).parent
E = here.parent
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
pp8 = Deployment(prefill_ep=8, decode_ep=8, prefill_pp=2, decode_pp=2, prefill_budget=2048)
ep16 = Deployment(prefill_budget=2048)
CASES = [   # name, deployment, (requests, rate, isl, osl, seed), reference, reference wall s (if recorded)
    ("E0031 1x8K/64 EP16", ep16, (1, 1.0, 8192, 64, 31), E / "E0031_expert_tp/tier2" / f"{ep16.key}.jsonl", 49.6),
    ("E0032 1x8K/64 PP2xEP8", pp8, (1, 1.0, 8192, 64, 31), E / "E0032_pipeline/tier2" / f"{pp8.key}.jsonl", 37.0),
    ("E0033 burst 8x2K PP1", Deployment(prefill_budget=2048), (8, 1000.0, 2048, 2, 33),
     E / "E0033_pp_overlap/tier2" / f"{Deployment(prefill_budget=2048).key}.jsonl", None),
    ("E0034 A EP16", ep16, RUN_A, E / "E0034_pp2_ep8_surrogates/tier2" / f"{ep16.key}.jsonl", None),
    ("E0034 A PP2xEP8", pp8, RUN_A, E / "E0034_pp2_ep8_surrogates/tier2" / f"{pp8.key}.jsonl", None),
    ("E0035 B PP2xEP8", pp8, RUN_B, E / "E0035_pp2_ep8_calibration/tier2" / f"{pp8.key}.jsonl", None),
]
for name, (tp, ep, pp) in (("tp4-ep4-pp4", (4, 4, 4)), ("tp8-ep8-pp2", (8, 8, 2))):
    d = layout_dep("h200", tp, ep, pp)
    CASES += [(f"E0036 {r} {name}", d, run, E / "E0036_layout_surrogates" / name / "tier2" / f"{r}-{d.key}.jsonl", None)
              for r, run in (("A", RUN_A), ("B", RUN_B))]
d128 = layout_dep("h200", 4, 16, 1, decode_batch=128)
CASES += [("E0037 A batch<=128", d128, (64, 40.0, 512, 32, 37), E / "E0037_decode_batch/tier2" / f"A-{d128.key}.jsonl", None)]


def stats(path):
    rows = [json.loads(l)["sub_requests"][0]["token_latency"] for l in Path(path).read_text().splitlines()]
    ttft = [r[0] for r in rows]
    tpot = [statistics.fmean(r[1:]) for r in rows if len(r) > 1]
    return {"ttft_p50": statistics.median(ttft), "ttft_max": max(ttft),
            "tpot_p50": statistics.median(tpot) if tpot else None, "n": len(rows)}


def grouped(case):
    name, dep, (n, rate, isl, osl, seed), ref, ref_wall = case
    out = here / "tier2" / f"{name.replace(' ', '_').replace('/', '-')}.jsonl"
    wall = None
    if not out.exists():
        cfg = copy.deepcopy(tier2_config(dep))
        for side in ("prefill_instances", "decode_instances"):
            for leaf in cfg["system"][side]:
                leaf["model_runner"]["lowering"] = "grouped"
        with tempfile.TemporaryDirectory(prefix="e0039-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(workloads.fixed(n, rate, isl, osl, seed=seed), tmp / "w.jsonl")
            t0 = time.perf_counter()
            subprocess.run([str(BIN), "--config", str(tmp / "c.toml"), "--workload", str(wl),
                            "--output-jsonl", str(tmp / "r.jsonl")], check=True, capture_output=True)
            wall = time.perf_counter() - t0
            out.parent.mkdir(exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
            out.with_suffix(".wall").write_text(f"{wall:.1f}\n")
    wall = float(out.with_suffix(".wall").read_text())
    a, b = stats(ref), stats(out)
    delta = {k: (b[k] / a[k] - 1) if a[k] else None for k in ("ttft_p50", "ttft_max", "tpot_p50")}
    return {"case": name, "request": a, "grouped": b, "delta": delta, "wall_grouped_s": wall, "wall_request_s": ref_wall}


if __name__ == "__main__":
    for c in CASES:
        assert c[3].exists(), c[3]
        tier2_config(c[1])
    with ThreadPoolExecutor(len(CASES)) as pool:
        rows = list(pool.map(grouped, CASES))
    for r in rows:
        d = r["delta"]
        print(f"  {r['case']:28s} TTFT p50 {d['ttft_p50']:+6.1%}  max {d['ttft_max']:+6.1%}  "
              + (f"TPOT p50 {d['tpot_p50']:+6.1%}" if d["tpot_p50"] is not None else "TPOT -      ")
              + f"  wall {r['wall_grouped_s']:6.0f} s" + (f" (was {r['wall_request_s']:.0f} s)" if r["wall_request_s"] else ""))
    (here / "result.json").write_text(json.dumps({"experiment": "E0039", "rows": rows, "source_identity": source_identity(),
                                                  "cost_model_version": cost_model_version()}, indent=1))
