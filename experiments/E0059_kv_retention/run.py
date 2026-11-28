"""E0059: A-E evaluation of M-KVRETAIN (mechanisms/specs/M-KVRETAIN.toml), Tier-1.

Runs against the tree named by INFEREVO_SIMTHESIZER (the candidate). Agent
workload (8-turn coding sessions, context 2k -> ~20k), 128 GPUs, 600 s probes,
capacity in sessions/s.
  A/B  agent optima without retention (E0058): H200 P5 D3 b64 mc8 rd,
       B300 P6 D2 b128 mc8
  C    same deployments with kv_retention
  D    co-search of retention with the P:D split (P3..P6)
  E    controls: (1) exercised: one session at low load, per-turn TTFT with vs
       without retention (turn 1 equal, later turns lower); (2) identity
       without streams: fixed 8K/1K run with the retention config is byte-
       identical; (3) memory pressure: prefill KV headroom cut to 2 GB/GPU
       (must evict: TTFT strictly above `on`) stays at or below
       no retention at the same memory (+5 %); (4) no `invalid` (zero-gap)
       probes.
Verdict VERIFIED iff C gains >= 5 % on some hardware and all controls pass.
`--replay` re-runs control (2)'s retention run and compares bytes.
"""

import hashlib
import json
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from adapters import steady, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from agent import search
from agent.controller import models_for

HERE = Path(__file__).parent


def dep(hw, p, batch, rd=False, retention=False):
    return Deployment(hardware=hw, prefill_instances=p, decode_instances=8 - p, prefill_ep=4, prefill_pp=4,
                      prefill_budget=2048, prefill_moe_chunks=8, prefill_dispatch="ranks" if rd else "leader",
                      decode_budget=batch, decode_max_batch=batch, kv_retention=retention)


BASE = {"h200": dep("h200", 5, 64, rd=True), "b300": dep("b300", 6, 128)}


def capacity(d):
    pb = search.Problem(hardware=d.hardware, workload="agent", duration=600.0)
    if not search.legal(pb, d):
        return None
    e = search.evaluate(pb, d)
    return {k: e[k] for k in ("capacity_rps", "tokens_per_s_gpu", "jct_p50_s", "jct_p99_s", "ttft_p99_ms",
                              "tpot_p50_ms", "bottleneck", "evidence")}


def config(d, agent, mem_util=None):
    """Tier-1 config; agent sessions run through the agent_dag front end (D20)."""
    cfg = te.tier1_config(tier2_config(d), models_for(d), retention=d.kv_retention)
    if mem_util is not None:
        for leaf in cfg["system"]["prefill_instances"]:
            leaf["scheduler"]["device_mem_utilization"] = mem_util
    return dict(cfg, system={"kind": "agent_dag", "inner": cfg["system"]}) if agent else cfg


def run_once(d, rows, mem_util=None):
    cfg = config(d, any(len(r["sub_requests"]) > 1 for r in rows), mem_util)
    with tempfile.TemporaryDirectory(prefix="e0059-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        te.run(tmp / "c.toml", workloads.write(rows, tmp / "w.jsonl"), tmp / "r.jsonl")
        data = (tmp / "r.jsonl").read_bytes()
        calls = steady.calls(tmp / "r.jsonl")
    return hashlib.sha256(data).hexdigest(), calls


def exercised(hw):
    """One 8-turn session alone: per-turn TTFT without / with retention."""
    out = {}
    for retention in (False, True):
        rows = workloads.agent_sessions(1, 8, 0.1, seed=59, streams=retention)
        _, calls = run_once(replace(BASE[hw], kv_retention=retention), rows)
        out[retention] = [round(c["ttft"] * 1e3, 1) for c in sorted(calls, key=lambda c: c["arrival"])]
    off, on = out[False], out[True]
    ok = abs(on[0] - off[0]) <= 1e-6 * max(1.0, off[0]) and all(b < a for a, b in zip(off[1:], on[1:]))
    return {"ttft_ms_off": off, "ttft_ms_on": on, "ok": ok}


def identity_without_streams(hw):
    rows = workloads.fixed(200, 4.0, 8192, 256, seed=59)
    a, _ = run_once(BASE[hw], rows)
    b, _ = run_once(replace(BASE[hw], kv_retention=True), rows)
    return {"off_sha256": a, "on_sha256": b, "ok": a == b}


def low_memory_utilization(hw):
    """Scheduler utilisation leaving a prefill GPU 2 GB above its weights:
    retained sessions (~3 GB/GPU at the test load on H200, more on B300) no
    longer fit, so the cache evicts; active requests are still admitted.
    Absolute, because B300's 161 GB of headroom defeats a relative cut."""
    from validation.capacity import Layout, evaluate
    r = evaluate(Layout(hw, 4, 4, 4), contexts=(20480,))
    w, usable = r.weight_bytes_per_gpu, r.usable_bytes_per_gpu
    return 0.9 * (w + 2e9) / usable


def pressure(hw, rate):
    """Retention under low prefill memory: steady metrics at a fixed rate."""
    out = {}
    low = low_memory_utilization(hw)
    for name, d, util in (("off", BASE[hw], None), ("on", replace(BASE[hw], kv_retention=True), None),
                          ("off_low_mem", BASE[hw], low), ("on_low_mem", replace(BASE[hw], kv_retention=True), low)):
        rows = workloads.agent_sessions(max(150, round(rate * 600)), 8, rate, seed=1, streams=d.kv_retention)
        cfg = config(d, True, util)
        with tempfile.TemporaryDirectory(prefix="e0059-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            te.run(tmp / "c.toml", workloads.write(rows, tmp / "w.jsonl"), tmp / "r.jsonl")
            m = steady.metrics(tmp / "r.jsonl", d.gpus, {"ttft_s": 5.0, "tpot_s": 0.05})
        out[name] = {"ttft_p50_ms": m["ttft_ms"]["p50"], "ttft_p99_ms": m["ttft_ms"]["p99"],
                     "jct_p50_s": m["jct_s"]["p50"], "queue": m["queue"], "slo_attainment": m["slo_attainment"]}
    t = {k: v["ttft_p50_ms"] for k, v in out.items()}
    # exercised: low memory must cost retention something (evictions); and at
    # equal memory retention must not be worse than none (graceful degradation)
    ok = (t["on"] < t["on_low_mem"] <= t["off_low_mem"] * 1.05
          and all(v["queue"] != "invalid" for v in out.values()))
    return dict(out, ok=ok)


def main(replay=False):
    if replay:
        stored = json.loads(max(HERE.glob("result-*.json"), key=lambda p: p.stat().st_mtime).read_text())
        got = identity_without_streams("h200")["on_sha256"]
        ok = got == stored["controls"]["identity_without_streams"]["h200"]["on_sha256"]
        print(json.dumps({"verdict": "REPRODUCED" if ok else "DIFFERS"}))
        return
    cases = {f"{hw} base": d for hw, d in BASE.items()}
    cases |= {f"{hw} retention": replace(d, kv_retention=True) for hw, d in BASE.items()}
    cases |= {f"h200 retention P{p}": replace(dep("h200", p, 64, rd=True), kv_retention=True) for p in (3, 4, 6)}
    cases |= {f"b300 retention P{p}": replace(dep("b300", p, 128), kv_retention=True) for p in (3, 4, 5)}
    for d in cases.values():
        tier2_config(d)
    with ProcessPoolExecutor(len(cases) + 6, mp_context=multiprocessing.get_context("fork")) as pool:
        caps = {n: pool.submit(capacity, d) for n, d in cases.items()}
        ex = {hw: pool.submit(exercised, hw) for hw in BASE}
        ident = {hw: pool.submit(identity_without_streams, hw) for hw in BASE}
        caps = {n: f.result() for n, f in caps.items()}
        ex = {hw: f.result() for hw, f in ex.items()}
        ident = {hw: f.result() for hw, f in ident.items()}
    press = {hw: pressure(hw, caps[f"{hw} base"]["capacity_rps"]) for hw in BASE}
    gain = {hw: caps[f"{hw} retention"]["capacity_rps"] / caps[f"{hw} base"]["capacity_rps"] - 1 for hw in BASE}
    best = {hw: max((n for n in caps if n.startswith(hw) and "retention" in n and caps[n]),
                    key=lambda n: caps[n]["capacity_rps"]) for hw in BASE}
    invalid = [n for n, c in caps.items() if c and (c["evidence"] or {}).get("queue") == "invalid"]
    controls_ok = (all(e["ok"] for e in ex.values()) and all(i["ok"] for i in ident.values())
                   and all(p["ok"] for p in press.values()) and not invalid)
    ok = controls_ok and max(gain.values()) >= 0.05
    for n, c in caps.items():
        print(f"{n:22s} " + ("n/a" if c is None else
              f"{c['capacity_rps']:5.2f} sess/s  {c['tokens_per_s_gpu']:5.1f} tok/s/GPU  JCT p50 {c['jct_p50_s']:.0f} s  "
              f"TTFT p99 {c['ttft_p99_ms']:.0f} ms  ({c['bottleneck']})"), flush=True)
    print("gain at the agent optimum:", {hw: f"{g:+.1%}" for hw, g in gain.items()}, "| best with retention:",
          {hw: f"{n} {caps[n]['capacity_rps']:.2f}" for hw, n in best.items()})
    for hw in BASE:
        print(hw, "per-turn TTFT ms off", ex[hw]["ttft_ms_off"], "on", ex[hw]["ttft_ms_on"],
              "| no-stream identity", ident[hw]["ok"], "| pressure", {k: (round(v["ttft_p50_ms"]), v["queue"])
              for k, v in press[hw].items() if k != "ok"})
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0059", "mechanism": "M-KVRETAIN", "tier": 1, "cost_model_version": cmv,
         "source_identity": source_identity(), "capacity": caps, "gain": gain, "best": best,
         "controls": {"exercised": ex, "identity_without_streams": ident, "pressure": press, "invalid": invalid}},
        indent=1, default=str))
    print(json.dumps({"verdict": "VERIFIED" if ok else "EXPLORATORY" if controls_ok else "FAILED",
                      "gain": gain, "controls_ok": controls_ok}))


if __name__ == "__main__":
    main(replay="--replay" in sys.argv)
