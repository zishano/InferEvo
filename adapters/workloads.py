"""Load-scaled workloads derived from a canonical Simthesizer workload.

`scaled` keeps each request's token shape and the trace's arrival pattern,
dividing arrival offsets by `rate_multiplier` (2.0 = twice the request rate).
`poisson` instead draws exponential inter-arrivals at `rate` req/s (seeded).
Output rows are canonical workload JSONL (one request DAG per row).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from adapters.engines import SIMTHESIZER

REALISTIC = SIMTHESIZER / "examples" / "workloads" / "realistic.jsonl"


def load(path: Path = REALISTIC) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scaled(rate_multiplier: float, source: Path = REALISTIC, start: float = 1.0) -> list[dict]:
    if rate_multiplier <= 0:
        raise ValueError("rate_multiplier must be positive")
    rows = sorted(load(source), key=lambda r: r["arrival_time"])
    t0 = rows[0]["arrival_time"]
    return [dict(r, arrival_time=start + (r["arrival_time"] - t0) / rate_multiplier) for r in rows]


def poisson(rate: float, seed: int = 0, source: Path = REALISTIC, start: float = 1.0) -> list[dict]:
    if rate <= 0:
        raise ValueError("rate must be positive")
    rng = random.Random(seed)
    rows, t = [], start
    for r in sorted(load(source), key=lambda r: r["arrival_time"]):
        rows.append(dict(r, arrival_time=t))
        t += rng.expovariate(rate)
    return rows


def offered_rate(rows: list[dict]) -> float:
    times = sorted(r["arrival_time"] for r in rows)
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0


def write(rows: list[dict], path: Path) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def agent_sessions(sessions: int, turns: int, rate: float, seed: int = 0,
                   system_prompt: int = 2000, user_tokens=(50, 400),
                   output_tokens=(50, 400), tool_output=(100, 2000),
                   tool_seconds=(0.2, 3.0), think_seconds=(0.0, 0.5),
                   start: float = 1.0, streams: bool = False) -> list[dict]:
    """Closed-loop coding-agent sessions (one request DAG per session).

    Turn k: LLM call whose input is the whole history (system prompt, prior
    user/tool tokens and model outputs), then a tool call whose output joins
    the history; the next LLM call is released only after the tool finishes.
    Sessions arrive as a seeded Poisson process at `rate` sessions/s.
    Synthetic: token/time ranges are assumptions, not a measured trace.
    `streams`: each LLM call declares `token_stream` = session id (its input
    is positions [0, input) of the session's token stream), which prefix
    caching / KV retention keys on (M-KVRETAIN); engines before it reject it.
    """
    rng = random.Random(seed)
    rows, t = [], start
    for s in range(sessions):
        history = system_prompt + rng.randint(*user_tokens)
        subs = []
        for k in range(turns):
            out = rng.randint(*output_tokens)
            last = k == turns - 1
            subs.append({"kind": "llm", "input_tokens": history, "output_tokens": out,
                         "next": [] if last else [len(subs) + 1],
                         **({"interval": rng.uniform(*think_seconds)} if k else {}),
                         **({"token_stream": s} if streams else {})})
            history += out
            if not last:
                tool_out = rng.randint(*tool_output)
                subs.append({"kind": "tool_call", "input_tokens": out, "output_tokens": tool_out,
                             "duration": rng.uniform(*tool_seconds), "next": [len(subs) + 1]})
                history += tool_out
        rows.append({"request_id": s, "arrival_time": t, "initial": [0], "sub_requests": subs})
        t += rng.expovariate(rate)
    return rows


def agent_max_context(turns: int, system_prompt: int = 2000, user_tokens=(50, 400),
                      output_tokens=(50, 400), tool_output=(100, 2000)) -> int:
    """Largest LLM-call context agent_sessions can produce (input + output)."""
    return system_prompt + user_tokens[1] + turns * output_tokens[1] + (turns - 1) * tool_output[1]


def tiled(copies: int, rate: float, seed: int = 0, source: Path = REALISTIC,
          start: float = 1.0) -> list[dict]:
    """`copies` repetitions of the source requests (token shapes kept, fresh
    request ids, order shuffled with a seed) arriving as a Poisson process at
    `rate` req/s. For scale studies: copies and rate proportional to the
    instance count keep per-instance offered load fixed."""
    if copies < 1 or rate <= 0:
        raise ValueError("copies >= 1 and rate > 0 required")
    rng = random.Random(seed)
    base = load(source)
    rows = [dict(r, request_id=c * len(base) + i) for c in range(copies) for i, r in enumerate(base)]
    rng.shuffle(rows)
    t = start
    for r in rows:
        r["arrival_time"] = t
        t += rng.expovariate(rate)
    return rows


def long_input(count: int, rate: float, seed: int = 0, input_range=(4096, 32768),
               output_range=(100, 1000), start: float = 1.0) -> list[dict]:
    """Long-prompt requests (RAG / document / code-context style): input
    log-uniform in input_range, output uniform in output_range, Poisson
    arrivals at `rate` req/s. Synthetic: ranges are assumptions."""
    import math
    rng = random.Random(seed)
    lo, hi = map(math.log, input_range)
    rows, t = [], start
    for i in range(count):
        rows.append({"request_id": i, "arrival_time": t, "initial": [0], "sub_requests": [{
            "kind": "llm", "input_tokens": int(math.exp(rng.uniform(lo, hi))),
            "output_tokens": rng.randint(*output_range), "next": []}]})
        t += rng.expovariate(rate)
    return rows


def fixed(count: int, rate: float, isl: int, osl: int, seed: int = 0, start: float = 1.0) -> list[dict]:
    """`count` requests of exactly isl input / osl output tokens, Poisson at
    `rate` req/s (benchmark-style fixed ISL/OSL)."""
    rng = random.Random(seed)
    rows, t = [], start
    for i in range(count):
        rows.append({"request_id": i, "arrival_time": t, "initial": [0], "sub_requests": [
            {"kind": "llm", "input_tokens": isl, "output_tokens": osl, "next": []}]})
        t += rng.expovariate(rate)
    return rows
