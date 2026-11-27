"""Reconstruct decode rounds of a P/D run from per-token result timings.

On a decode instance every round emits one token for each request in its
batch, so requests whose output tokens share an emission time form one round.
For output index i >= 1 the round processed one query token with
kv_len = input_tokens + i - 1 cached tokens. Duration = gap to the previous
emission time when the instance was continuously busy (some request
continues from the previous round). Rounds that contain a request's first
decode token also wait for the P->D state handoff and are flagged.
Valid for one decode instance; multi-instance runs need instance attribution.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DecodeRound:
    time_s: float
    requests: list[int]
    kv_lens: list[int]
    duration_s: float | None     # None when the instance was idle before it
    first_decode: bool


def reconstruct(result_jsonl: Path, digits: int = 12) -> list[DecodeRound]:
    emit = defaultdict(list)
    for line in result_jsonl.read_text().splitlines():
        row = json.loads(line)
        (sub,) = row["sub_requests"]
        t, inp = row["arrival_time"], sub["input_tokens"]
        for i, gap in enumerate(sub["token_latency"]):
            t += gap
            if i >= 1:
                emit[round(t, digits)].append((row["request_id"], inp + i - 1, i == 1))
    rounds, previous = [], set()
    times = sorted(emit)
    for k, t in enumerate(times):
        members = sorted(emit[t])
        ids = {m[0] for m in members}
        busy = k > 0 and bool(ids & previous)
        rounds.append(DecodeRound(
            t, [m[0] for m in members], [m[1] for m in members],
            t - times[k - 1] if busy else None, any(m[2] for m in members)))
        previous = ids
    return rounds


def reconstruct_multi(result_jsonl: Path, digits: int = 12) -> dict[int, list[DecodeRound]]:
    """Multi-instance version: a request decodes on one instance, and a round
    belongs to one instance, so requests that ever share a round are on the
    same instance (union-find over co-membership). Rounds are then chained
    per instance; durations are gaps within the instance only."""
    emit = defaultdict(list)
    for line in result_jsonl.read_text().splitlines():
        row = json.loads(line)
        (sub,) = row["sub_requests"]
        t, inp = row["arrival_time"], sub["input_tokens"]
        for i, gap in enumerate(sub["token_latency"]):
            t += gap
            if i >= 1:
                emit[round(t, digits)].append((row["request_id"], inp + i - 1, i == 1))
    parent: dict[int, int] = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for members in emit.values():
        first = find(members[0][0])
        for rid, _, _ in members[1:]:
            parent[find(rid)] = first
    by_comp = defaultdict(list)
    for t in sorted(emit):
        by_comp[find(emit[t][0][0])].append(t)
    out = {}
    for n, (comp, times) in enumerate(sorted(by_comp.items(), key=lambda kv: kv[1][0])):
        rounds, previous = [], set()
        for k, t in enumerate(times):
            members = sorted(emit[t])
            ids = {m[0] for m in members}
            busy = k > 0 and bool(ids & previous)
            rounds.append(DecodeRound(t, [m[0] for m in members], [m[1] for m in members],
                                      t - times[k - 1] if busy else None, any(m[2] for m in members)))
            previous = ids
        out[n] = rounds
    return out
