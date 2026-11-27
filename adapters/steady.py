"""Steady-state serving metrics and SLO capacity search.

Burst runs (all requests arrive in a few seconds) measure drain speed; here
metrics cover only calls that *arrive* inside a steady window (between the
`warm` and `1 - cool` quantiles of call arrival times), and throughput
counts tokens *emitted* inside that window. Capacity = the highest offered
rate at which >= `target` of window calls meet the SLO and no queue is
growing: prefill (TTFT p50 of the window's last third <= 2x its first third,
or below the TTFT SLO's 1/4) and decode (decode time = e2e - TTFT, last
third p50 <= 1.25x first third). The decode check catches P/D backlog where
calls wait for a decode slot: that wait is one inter-token gap, invisible in
mean TPOT. A run whose calls emit tokens with zero gaps (> 1 % of gaps;
SIM-DEFECT-3: preempted P/D decode requests in the legacy scheduler) is
invalid, never a capacity point.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def calls(result_jsonl: Path) -> list[dict]:
    """Per LLM call: arrival (start of the call in its chain), input,
    token emission times, ttft, mean tpot."""
    out = []
    for line in Path(result_jsonl).read_text().splitlines():
        r = json.loads(line)
        t = r["arrival_time"]
        for s in r["sub_requests"]:
            t += s.get("interval") or 0.0
            if s["kind"] != "llm":
                t += s["duration"]
                continue
            lat, emit, e = s["token_latency"], [], t
            for gap in lat:
                e += gap
                emit.append(e)
            out.append({"arrival": t, "input": s["input_tokens"], "emit": emit, "ttft": lat[0],
                        "zero_gaps": sum(gap <= 1e-9 for gap in lat[1:]), "gaps": len(lat) - 1,
                        "tpot": statistics.fmean(lat[1:]) if len(lat) > 1 else 0.0})
            t = e
    return out


def _q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def metrics(result_jsonl: Path, gpus: int, slo: dict, warm: float = 0.1, cool: float = 0.1,
            sides: tuple[int, int] | None = None) -> dict:
    """`sides` = (prefill GPUs, decode GPUs) adds per-side throughput: prefill =
    input tokens whose prefill (first token) completes in the window, decode =
    output tokens emitted in the window, each per GPU of its own side."""
    cs = calls(result_jsonl)
    arrivals = sorted(c["arrival"] for c in cs)
    w0, w1 = _q(arrivals, warm), _q(arrivals, 1 - cool)
    win = sorted((c for c in cs if w0 <= c["arrival"] <= w1), key=lambda c: c["arrival"])
    ok = [c["ttft"] <= slo["ttft_s"] and c["tpot"] <= slo["tpot_s"] for c in win]
    span = w1 - w0
    emitted = sum(w0 <= e <= w1 for c in cs for e in c["emit"])
    good = sum(w0 <= e <= w1 for c, g in zip(win, ok) if g for e in c["emit"])
    third = max(1, len(win) // 3)
    early = statistics.median(c["ttft"] for c in win[:third])
    late = statistics.median(c["ttft"] for c in win[-third:])
    decode = lambda part: statistics.median(c["emit"][-1] - c["arrival"] - c["ttft"] for c in part)
    prefill_queue = not (late <= 2 * early or late <= slo["ttft_s"] / 4)
    decode_growth = decode(win[-third:]) / decode(win[:third]) if decode(win[:third]) > 0 else 1.0
    zero = sum(c["zero_gaps"] for c in win) / max(1, sum(c["gaps"] for c in win))
    prefilled = sum(c["input"] for c in cs if c["emit"] and w0 <= c["emit"][0] <= w1)
    queue = ("invalid" if zero > 0.01 else "prefill" if prefill_queue
             else "decode" if decode_growth > 1.25 else None)
    return {"window_s": span, "window_calls": len(win), "offered_rate": (len(win) - 1) / span,
            "slo": slo, "slo_attainment": sum(ok) / len(ok),
            # which SLO fails (diagnosis: TTFT -> prefill side, TPOT -> decode side)
            "ttft_violation": sum(c["ttft"] > slo["ttft_s"] for c in win) / len(win),
            "tpot_violation": sum(c["tpot"] > slo["tpot_s"] for c in win) / len(win),
            "tokens_per_s_gpu": emitted / span / gpus,
            "prefill_tokens_per_s": prefilled / span, "decode_tokens_per_s": emitted / span,
            **({"prefill_tokens_per_s_gpu": prefilled / span / sides[0],
                "decode_tokens_per_s_gpu": emitted / span / sides[1]} if sides else {}),
            "goodput_tokens_per_s_gpu": good / span / gpus,
            "ttft_ms": {"p50": _q([c["ttft"] for c in win], .5) * 1e3, "p99": _q([c["ttft"] for c in win], .99) * 1e3},
            "tpot_ms": {"p50": _q([c["tpot"] for c in win], .5) * 1e3, "p99": _q([c["tpot"] for c in win], .99) * 1e3},
            # per-user decode speed (aiconfigurator-style tokens/s/user): 1 / TPOT
            "tokens_per_s_user": {"p50": 1 / _q([c["tpot"] for c in win if c["tpot"] > 0], .5),
                                  "p10": 1 / _q([c["tpot"] for c in win if c["tpot"] > 0], .9)},
            "queue_growth": late / early if early else None, "decode_growth": decode_growth,
            "zero_gap_fraction": zero,
            "queue": queue, "stable": queue is None}


def meets(m: dict, target: float = 0.9) -> bool:
    return m["stable"] and m["slo_attainment"] >= target


def capacity(run, lo: float, hi: float, target: float = 0.9, steps: int = 7) -> dict:
    """Bisect (geometric) the highest rate with meets(run(rate)). `run(rate)`
    returns metrics. Expands `hi` if it still meets, shrinks `lo` if it fails."""
    seen = {}
    def ok(rate):
        seen[rate] = run(rate)
        return meets(seen[rate], target)
    while ok(hi) and hi < lo * 1e4:
        lo, hi = hi, hi * 2
    while not ok(lo) and lo > 1e-4:
        hi, lo = lo, lo / 2
    if not meets(seen[lo], target):
        return {"rate": 0.0, "probes": seen}
    for _ in range(steps):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return {"rate": lo, "at_rate": seen[lo], "probes": {f"{k:.4g}": v for k, v in seen.items()}}
