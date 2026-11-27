"""LLM-agent policy for the research controller (Phase 3).

Same interface as `rule_policy`: policy(entry) -> [(Deployment, rationale)].
The agent sees the ledger so far (deployments, objectives, SLO, bottleneck
labels with evidence) and a numbered list of candidate moves from the
current entry (diagnosis moves + one-step neighbours); it may only choose
among those options, so every proposal stays inside validated surrogate
coverage. It answers JSON: [{"option": i, "rationale": "..."}] in priority
order. Unparseable answers or out-of-range options fall back to the rules
(recorded in the rationale).

`complete(prompt) -> str` is injected: no provider SDK or network call
lives here. Sending ledgers to an external model service is an outward
action and needs the user's go-ahead.
"""

from __future__ import annotations

import json
import re

from adapters.topologies import Deployment
from agent.controller import neighbours, rule_policy

SYSTEM = """You are the research agent of an LLM-serving simulation platform.
Goal: maximize SLO goodput (tokens/s/GPU of requests meeting the SLO) and keep
the Pareto set of (throughput up, TTFT p99 down, TPOT p50 down) well covered,
with as few evaluations as possible. Use the bottleneck evidence. Choose only
from the numbered options. Answer with JSON only:
[{"option": <number>, "rationale": "<one sentence citing evidence>"}]"""


def _summary(e: dict) -> dict:
    o = e["objectives"]
    return {"deployment": e["key"], "goodput": round(o.get("goodput_tokens_per_s_gpu") or 0, 2),
            "tok_s_gpu": round(o["tokens_per_s_gpu"], 2), "ttft_p99_ms": round(o["ttft_p99_ms"]),
            "tpot_p50_ms": round(o["tpot_p50_ms"] or 0, 2),
            "bottlenecks": [{k: b[k] for k in ("class", "evidence") if k in b} for b in e["bottlenecks"]],
            "rationale": e["rationale"]}


def prompt_for(entry: dict, ledger: list[dict], options: list[tuple[Deployment, str]]) -> str:
    lines = [SYSTEM, "", "Ledger (evaluated so far):"]
    lines += [json.dumps(_summary(e)) for e in ledger if "objectives" in e]
    lines += ["", f"Current: {entry['key']}", "Options:"]
    lines += [f"{i}: {d.key} -- {why}" for i, (d, why) in enumerate(options)]
    return "\n".join(lines)


def parse(answer: str, n_options: int) -> list[tuple[int, str]]:
    m = re.search(r"\[.*\]", answer, re.S)
    if not m:
        raise ValueError("no JSON list in answer")
    picks = []
    for item in json.loads(m.group(0)):
        i = int(item["option"])
        if not 0 <= i < n_options:
            raise ValueError(f"option {i} out of range")
        picks.append((i, str(item.get("rationale", ""))))
    return picks


def llm_policy(complete, controller):
    """Build a policy bound to a controller (for the ledger) and a model."""
    def policy(entry: dict) -> list[tuple[Deployment, str]]:
        dep = Deployment.of(entry["deployment"])
        options, seen = [], set()
        for d, why in rule_policy(entry) + neighbours(dep):
            if d.key not in seen and d.key not in controller.seen:
                options.append((d, why))
                seen.add(d.key)
        if not options:
            return []
        try:
            picks = parse(complete(prompt_for(entry, list(controller.seen.values()), options)), len(options))
        except Exception as error:      # any failure -> rules, recorded
            return [(d, f"{why} [llm fallback: {error}]") for d, why in rule_policy(entry)]
        return [(options[i][0], f"llm: {why}") for i, why in picks]
    return policy
