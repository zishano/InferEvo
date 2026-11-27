"""Cross-layer feedback record and bottleneck classification (Phase 3).

Builds one record per run from engine outputs: request/call level (TTFT,
TPOT), session level (agent JCT for DAG requests), instance level (busy
fraction, mean batch, tokens per step from the Tier-1 step trace), plus
classified bottlenecks with the evidence behind each label. Thresholds are
explicit parameters, not hidden heuristics.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

STEP = re.compile(r"StepModel\(instance=(\S+) seconds=([0-9.eE+-]+) requests=(\d+) tokens=(\d+)\)")


def _q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def requests(result_jsonl: Path, gpus: int, slo: dict | None = None) -> dict:
    """slo: {"ttft_s": .., "tpot_s": ..}; goodput counts tokens of calls that
    meet both (spec 11: SLO goodput)."""
    ttft, tpot, users, jct, tokens, calls = [], [], [], [], 0, []
    good_tokens = 0
    first_arrival, last_finish = float("inf"), 0.0
    for line in Path(result_jsonl).read_text().splitlines():
        r = json.loads(line)
        t = r["arrival_time"]
        first_arrival = min(first_arrival, t)
        end = t
        for s in r["sub_requests"]:          # chains: completion = sum along the chain
            end += (s.get("interval") or 0.0)
            if s["kind"] == "llm":
                lat = s["token_latency"]
                ttft.append(lat[0])
                calls.append((r["request_id"], s["input_tokens"], lat[0]))
                if len(lat) > 1:
                    tpot.append(statistics.fmean(lat[1:]))
                    users.append((len(lat) - 1) / sum(lat[1:]))
                tokens += len(lat)
                end += sum(lat)
                mean_tpot = statistics.fmean(lat[1:]) if len(lat) > 1 else 0.0
                if slo and lat[0] <= slo["ttft_s"] and mean_tpot <= slo["tpot_s"]:
                    good_tokens += len(lat)
            else:
                end += s["duration"]
        jct.append(end - t)
        last_finish = max(last_finish, end)
    span = last_finish - first_arrival
    return {"calls": len(ttft), "sessions": len(jct),
            "ttft_ms": {"min": min(ttft) * 1e3, "p50": _q(ttft, .5) * 1e3, "p99": _q(ttft, .99) * 1e3},
            "tpot_ms": {"p50": _q(tpot, .5) * 1e3, "p99": _q(tpot, .99) * 1e3} if tpot else None,
            "jct_s": {"mean": statistics.fmean(jct), "p99": _q(jct, .99)},
            "tokens_per_s_gpu": tokens / span / gpus, "tokens_per_s_user": statistics.fmean(users) if users else None,
            "span_s": span, "_calls": calls,
            **({"slo": slo, "goodput_tokens_per_s_gpu": good_tokens / span / gpus,
                "slo_attainment": good_tokens / tokens} if slo else {})}


def instances(trace_jsonl: Path) -> dict:
    """Per-instance busy time from Tier-1 step nodes (creation + dispatch)."""
    steps, starts = {}, {}
    for line in Path(trace_jsonl).read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "node_create":
            m = STEP.search(str(e.get("details", "")))
            if m:
                steps[e["node"]] = (m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4)))
        elif e.get("event") == "dispatch":
            starts[e["node"]] = e["time_s"]
    per = defaultdict(lambda: {"steps": 0, "busy_s": 0.0, "requests": 0, "tokens": 0,
                               "first_s": float("inf"), "last_s": 0.0})
    for node, (inst, sec, req, tok) in steps.items():
        p = per[inst]
        start = starts.get(node)
        p["steps"] += 1; p["busy_s"] += sec; p["requests"] += req; p["tokens"] += tok
        if start is not None:
            p["first_s"] = min(p["first_s"], start); p["last_s"] = max(p["last_s"], start + sec)
    return {k: {"steps": v["steps"], "busy_s": v["busy_s"],
                "mean_batch": v["requests"] / v["steps"], "tokens_per_step": v["tokens"] / v["steps"],
                "window_s": v["last_s"] - v["first_s"]} for k, v in per.items()}


def classify(record: dict, max_batch: dict | None = None, busy_hi: float = 0.85,
             occupancy_hi: float = 0.85, queue_share_hi: float = 0.5) -> list[dict]:
    """Bottleneck labels with evidence.

    Prefill saturation: busy fraction over the instance's active window
    (first to last step) — prefill work is finite, so idle time inside the
    window is real slack; the post-arrival decode tail is excluded. Decode saturation: batch occupancy
    (mean batch / max_batch) — continuous batching keeps decode busy whenever
    any request is active, so busy fraction is not a saturation signal.
    """
    span = record["requests"]["span_s"]
    out = []
    p_busy = [v["busy_s"] / v["window_s"] for k, v in record.get("instances", {}).items()
              if k.startswith("p") and v["window_s"] > 0]
    if p_busy and max(p_busy) >= busy_hi:
        out.append({"class": "prefill_saturated", "evidence": {"p_busy": p_busy}})
    if max_batch:
        occ = {k: v["mean_batch"] / max_batch[k[0]] for k, v in record.get("instances", {}).items()
               if k.startswith("d")}
        if occ and max(occ.values()) >= occupancy_hi:
            out.append({"class": "decode_batch_saturated", "evidence": {"occupancy": occ}})
    for side in ("p", "d"):
        vals = [v["busy_s"] / v["window_s"] for k, v in record.get("instances", {}).items()
                if k.startswith(side) and v["window_s"] > 0]
        if len(vals) > 1 and max(vals) - min(vals) > 0.25:
            out.append({"class": f"{side}_instance_imbalance", "evidence": {"busy": vals}})
    q = record.get("queueing")
    if q and q["median_queue_share"] > queue_share_hi:
        out.append({"class": "ttft_queueing", "evidence": q})
    return out


def limits(config: dict) -> dict:
    """Instance-level max_batch per side from a P/D config (any wrapper depth)."""
    system = config["system"]
    while system.get("kind") == "agent_dag":
        system = system["inner"]
    ep = lambda leaf: leaf["model_runner"].get("ep", 1) if leaf["model_runner"]["kind"] == "step_model" else 1
    out = {}
    for side, key in (("p", "prefill_instances"), ("d", "decode_instances")):
        leaf = system[key][0]
        out[side] = leaf["scheduler"]["max_batch"] * ep(leaf)
    return out


def queueing(calls, prefill_model, budget: int) -> dict:
    """Share of TTFT not explained by the call's own isolated prefill service
    (sum of its chunk rounds under the prefill surrogate)."""
    shares = []
    for rid, tokens, ttft in calls:
        service = sum(prefill_model.step_time([(rid, min(budget, tokens - d), d)])
                      for d in range(0, tokens, budget))
        shares.append(max(0.0, ttft - service) / ttft)
    return {"median_queue_share": statistics.median(shares), "p90_queue_share": _q(shares, .9)}


def record(result_jsonl: Path, gpus: int, trace_jsonl: Path | None = None,
           config: dict | None = None, prefill_model=None, budget: int = 128,
           slo: dict | None = None) -> dict:
    rec = {"requests": requests(result_jsonl, gpus, slo)}
    calls = rec["requests"].pop("_calls")
    if prefill_model is not None:
        rec["queueing"] = queueing(calls, prefill_model, budget)
    if trace_jsonl:
        rec["instances"] = instances(trace_jsonl)
    rec["bottlenecks"] = classify(rec, limits(config) if config else None)
    return rec
