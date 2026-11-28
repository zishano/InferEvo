"""E0049: A-E evaluation of M-FP8LINEAR (mechanisms/specs/M-FP8LINEAR.toml).

Semantics-changing mechanism: performance is simulated, model quality is not
(quality status in the spec, UNVERIFIED). Runs against the tree named by
INFEREVO_SIMTHESIZER (the candidate). Tier-2.

  A  parent: linear = bf16 (official checkpoint precision)
  B  best existing decode configuration: EP16 TP4, max batch 64 (MC-5 / E0041)
  C  mechanism on the decode side: linear = fp8
  D  co-search with decode max batch (32 / 64) and the prefill side (fp8 on
     the fastest prefill configuration PP4xEP4 mc8 rank dispatch)
  E  controls: accounting on one decode round at the operating point (64
     calls, 8.7K context: converted GEMMs compute seconds x0.5 exactly, HBM
     down; every other kernel class identical), equal-concurrency guard

Decode case: P1 (PP4xEP4, mc8, rank dispatch, budget 2048) : D1 (EP16 TP4),
`batch` simultaneous calls of (64 in, 64 out); metric TPOT p50 at peak
concurrency == batch for both variants (r0 compared different concurrency). Prefill case: 8 x 8192 burst,
makespan. Verdict VERIFIED (performance) iff accounting holds and decode TPOT
at batch 64 drops >= 5 %. `--replay` re-runs the best decode case.
"""

import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters import steady, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, enable_bridge_imports, source_identity
from adapters.topologies import Deployment, tier2_config

HERE = Path(__file__).parent
OUT = HERE / "tier2"
FAST_PREFILL = {"kind": "grouped", "moe_chunks": 8, "dispatch": "ranks"}
# Decode at a fixed round size: `batch` calls with tiny prompts arrive at once,
# so every decode round holds `batch` calls in both variants (E0041 method; a
# prefill-limited burst let the faster variant decode at lower concurrency).
DECODE_WL = lambda batch: (batch, 1000.0, 64, 64, 49)
PREFILL_WL = (8, 1000.0, 8192, 2, 44)
CONVERTED = {"wq_a", "wq_b", "wkv", "kda_wq", "kda_wk", "kda_wv", "output_gate", "wo",
             "routed_up", "routed_down", "sw_up", "sw_down", "dense_up", "dense_down"}


def dep(batch=64):
    return Deployment(hardware="h200", prefill_ep=4, prefill_pp=4, prefill_budget=2048,
                      decode_budget=batch, decode_max_batch=batch)


def config(d, prefill_linear="bf16", decode_linear="bf16"):
    cfg = tier2_config(d)
    for side, linear, base in (("prefill", prefill_linear, FAST_PREFILL), ("decode", decode_linear, None)):
        for inst in cfg["system"][f"{side}_instances"]:
            opts = dict(base or {"kind": "grouped"}, **({"linear": linear} if linear != "bf16" else {}))
            inst["model_runner"]["lowering"] = opts if len(opts) > 1 else "grouped"
    return cfg


def run(name, cfg, wl_spec):
    out = OUT / f"{name}.jsonl"
    if not out.exists():
        with tempfile.TemporaryDirectory(prefix="e0049-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(workloads.fixed(*wl_spec[:4], seed=wl_spec[4]), tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(tmp / "r.jsonl")],
                           check=True, capture_output=True)
            OUT.mkdir(exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
    cs = steady.calls(out)
    events = sorted([(c["emit"][0], 1) for c in cs] + [(c["emit"][-1], -1) for c in cs])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    first = min(c["arrival"] for c in cs)
    tpot = sorted(c["tpot"] for c in cs)
    return {"makespan_s": max(c["emit"][0] for c in cs) - first,
            "tpot_p50_ms": statistics.median(tpot) * 1e3, "tpot_p90_ms": tpot[int(0.9 * len(tpot))] * 1e3,
            "decode_concurrency": peak, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def accounting():
    """Per node class seconds of one decode round (16 calls, 8.7K context), bf16 vs fp8."""
    enable_bridge_imports()
    from rooflang_bridge.kimi_k3_shared import KimiTemplate
    rows = [{"q_len": 1, "kv_len": 8704, "lm_head_len": 1}] * 64
    agg = {}
    for linear in ("bf16", "fp8"):
        lowering = "grouped" if linear == "bf16" else {"kind": "grouped", "linear": linear}
        t = KimiTemplate(4, 16, {"gpu_model": "h200", "gpu_count": 16}, 0, stage="d", lowering=lowering)
        a = defaultdict(lambda: defaultdict(float))
        for op in t.export(rows, list(range(64)), details=True)["operations"]:
            m = re.match(r"request\d+\.(?:layer\d+\.)?([a-z_]+)", op["work"].get("details", {}).get("name", ""))
            for d in op["work"]["demands"]:
                for r, x in d["resources"].items():
                    a[m.group(1) if m else "?"][r.split(":")[0]] += x
        agg[linear] = a
    out, ok = {}, True
    for cat in sorted(set(agg["bf16"]) | set(agg["fp8"])):
        b, f = agg["bf16"][cat], agg["fp8"][cat]
        ratio = {k: f.get(k, 0) / v for k, v in b.items() if v}
        if cat in CONVERTED:
            ok &= abs(ratio.get("compute", 0.5) - 0.5) < 1e-9 and ratio.get("hbm", 1) < 1
        else:
            ok &= all(abs(x - 1) < 1e-12 for x in ratio.values())
        out[cat] = ratio
    total = {lin: sum(v.get("hbm", 0) for v in agg[lin].values()) for lin in agg}
    return {"ok": ok, "ratios": out, "hbm_total_ratio": total["fp8"] / total["bf16"]}


def main(replay=False):
    if replay:
        best = json.loads(max(HERE.glob("result-*.json"), key=lambda p: p.stat().st_mtime).read_text())["best"]
        moved = OUT / f"{best['case']}.jsonl.replayed-from"
        (OUT / f"{best['case']}.jsonl").rename(moved)
        got = run(best["case"], config(dep(best["batch"]), decode_linear="fp8"), DECODE_WL(best["batch"]))["sha256"]
        moved.unlink()
        print(json.dumps({"verdict": "REPRODUCED" if got == best["sha256"] else "DIFFERS", "case": best["case"]}))
        return
    jobs = {f"d-b{b}-{lin}": (config(dep(b), decode_linear=lin), DECODE_WL(b)) for b in (32, 64) for lin in ("bf16", "fp8")}
    jobs |= {f"p-{lin}": (config(dep(), prefill_linear=lin), PREFILL_WL) for lin in ("bf16", "fp8")}
    with ThreadPoolExecutor(len(jobs) + 1) as pool:
        futs = {n: pool.submit(run, n, *a) for n, a in jobs.items()}
        acc = pool.submit(accounting)
        res = {n: f.result() for n, f in futs.items()}
        acc = acc.result()
    conc = {b: [res[f"d-b{b}-{lin}"]["decode_concurrency"] for lin in ("bf16", "fp8")] for b in (32, 64)}
    if any(c != [b, b] for b, c in conc.items()):
        raise SystemExit(f"decode rounds not at the target batch in both variants: {conc}")
    gain = {b: 1 - res[f"d-b{b}-fp8"]["tpot_p50_ms"] / res[f"d-b{b}-bf16"]["tpot_p50_ms"] for b in (32, 64)}
    prefill_gain = 1 - res["p-fp8"]["makespan_s"] / res["p-bf16"]["makespan_s"]
    ok = acc["ok"] and gain[64] >= 0.05
    best = {"case": "d-b64-fp8", "batch": 64, "sha256": res["d-b64-fp8"]["sha256"]}
    summary = {"A_parent": {k: v for k, v in res.items() if k.endswith("bf16")},
               "B_best_existing": res["d-b64-bf16"], "C_decode_fp8": res["d-b64-fp8"],
               "D_cosearch": {"decode_tpot_gain_by_batch": gain, "prefill_makespan_gain": prefill_gain},
               "E_controls": {"accounting": acc, "decode_concurrency": conc}}
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0049", "mechanism": "M-FP8LINEAR", "quality": "UNVERIFIED (spec [quality])",
         "cost_model_version": cmv, "source_identity": source_identity(), "runs": res, "summary": summary,
         "best": best}, indent=1))
    for n, r in sorted(res.items()):
        print(n, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items() if k != "sha256"})
    print("decode TPOT gain", {b: f"{g:+.1%}" for b, g in gain.items()}, "| prefill makespan gain", f"{prefill_gain:+.1%}",
          "| decode-round HBM ratio", round(acc["hbm_total_ratio"], 4), "| accounting ok", acc["ok"])
    print(json.dumps({"verdict": "VERIFIED" if ok else "EXPLORATORY" if acc["ok"] else "FAILED",
                      "quality": "UNVERIFIED", "best": best, "decode_gain": gain, "prefill_gain": prefill_gain}))


if __name__ == "__main__":
    main(replay="--replay" in sys.argv)
