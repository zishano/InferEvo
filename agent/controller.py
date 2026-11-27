"""Research controller v0: diagnosis-driven configuration search (Phase 3).

Loop: evaluate (Tier-1) -> feedback -> diagnose -> propose -> evaluate ...
Every evaluation is appended to a JSONL ledger with its rationale; a Pareto
archive keeps non-dominated deployments. `propose` is the pluggable policy:
the built-in rule policy maps bottleneck labels to moves; an LLM agent can
replace it by reading the ledger and returning Deployments with rationales.
Tier-1 results are screening evidence; archive members are confirmed at
Tier-2 separately (confirm()).
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from adapters import feedback, tier1, tier1_engine as te, workloads
from adapters.engines import ROOT
from adapters.topologies import Deployment, tier2_config

E = ROOT / "experiments"
OBJECTIVES = {"tokens_per_s_gpu": +1, "ttft_p99_ms": -1, "tpot_p50_ms": -1}
DEFAULT_SLO = {"ttft_s": 2.0, "tpot_s": 0.05}   # assumption; recorded per entry


# Tier-1 surrogates per side layout (hardware, attention TP, MoE ranks per
# stage, PP), fitted and load-calibrated under the reference grouped Tier-2
# lowering (MC-5, E0040). Fits from the per-request lowering (E0012-E0037)
# describe a different Tier-2 and are no longer used.
FITS = Path("E0040_grouped_surrogates")
# Prefill surrogates with expert-chunk pipelining (M-MOECHUNK), name suffix -mc<k>.
CHUNK_FITS = Path("E0045_moechunk_surrogates")
# A fitted decode model is used only if its held-out loaded TPOT error (run B,
# after calibration) is within this bound; otherwise the layout is prefill-only.
DECODE_HELDOUT_MAX = 0.05
# Decode models valid beyond 32 requests per round.
LARGE_DECODE = {("h200", 4, 16, 1): "E0041_grouped_decode_batch/result.json"}


def _validated_decode(fit: dict) -> dict | None:
    err = fit.get("runs", {}).get("B", {}).get("rel_err_calibrated", {}).get("tpot_p50")
    return fit["decode"] if err is None or abs(err) <= DECODE_HELDOUT_MAX else None


def _surrogates(hardware: str, tp: int, ep: int = 16, pp: int = 1, moe_chunks: int = 1):
    """(prefill model, decode model or None, validated budgets) for one side's layout."""
    name = f"{hardware}-tp{tp}-ep{ep}-pp{pp}" + (f"-mc{moe_chunks}" if moe_chunks > 1 else "")
    path = E / (FITS if moe_chunks == 1 else CHUNK_FITS) / name / "result.json"
    if not path.is_file():
        raise KeyError(f"no validated Tier-1 surrogate for {hardware} TP{tp} EP{ep} PP{pp}"
                       + (f" MoE chunks {moe_chunks}" if moe_chunks > 1 else ""))
    r = json.loads(path.read_text())
    return r["prefill"], _validated_decode(r), (128, 512, 2048)


def models_for(dep: Deployment) -> dict:
    """Per-side validated Tier-1 surrogates; KeyError outside their fitted range."""
    if (dep.prefill_etp, dep.decode_etp) != (1, 1):
        raise KeyError("Tier-1 surrogates fitted at ETP1 only")
    if dep.prefill_max_batch > 32:
        raise KeyError("prefill surrogates fitted at max batch <= 32")
    if dep.decode_moe_chunks != 1:
        raise KeyError("decode surrogates fitted without MoE chunking (E0044: no decode gain)")
    pre, _, budgets = _surrogates(dep.hardware, dep.prefill_tp, dep.prefill_ep, dep.prefill_pp,
                                  dep.prefill_moe_chunks)
    _, dec, _ = _surrogates(dep.hardware, dep.decode_tp, dep.decode_ep, dep.decode_pp)
    if dec is None:
        raise KeyError(f"decode surrogate for TP{dep.decode_tp} EP{dep.decode_ep} PP{dep.decode_pp} "
                       f"failed held-out validation (> {DECODE_HELDOUT_MAX:.0%} TPOT error)")
    if dep.decode_max_batch > dec["key"].get("max_requests", 32):
        large = E / LARGE_DECODE.get((dep.hardware, dep.decode_tp, dep.decode_ep, dep.decode_pp), "-")
        dec = _validated_decode(json.loads(large.read_text())) if large.is_file() else None
        if dec is None or dep.decode_max_batch > dec["key"]["max_requests"]:
            raise KeyError(f"no decode surrogate validated for max batch {dep.decode_max_batch}")
    if dep.prefill_budget not in budgets:
        raise KeyError(f"prefill surrogate not validated for budget {dep.prefill_budget}")
    return {"p": tier1.Tier1Model(**pre), "d": tier1.Tier1Model(**dec)}


def objectives(rec: dict) -> dict:
    r = rec["requests"]
    return {"tokens_per_s_gpu": r["tokens_per_s_gpu"], "ttft_p99_ms": r["ttft_ms"]["p99"],
            "tpot_p50_ms": r["tpot_ms"]["p50"] if r["tpot_ms"] else None,
            "goodput_tokens_per_s_gpu": r.get("goodput_tokens_per_s_gpu"),
            "slo_attainment": r.get("slo_attainment")}


def dominates(a: dict, b: dict) -> bool:
    better = False
    for k, sign in OBJECTIVES.items():
        if a[k] is None or b[k] is None:
            continue
        if sign * (a[k] - b[k]) < 0:
            return False
        better |= sign * (a[k] - b[k]) > 0
    return better


class Controller:
    def __init__(self, ledger: Path, workload_rows: list[dict], workload_label: str,
                 slo: dict | None = None):
        self.slo = slo or DEFAULT_SLO
        self.ledger = ledger
        self.rows = workload_rows
        self.label = workload_label
        self.archive: list[dict] = []
        self.seen: dict[str, dict] = {}

    def evaluate(self, dep: Deployment, rationale: str) -> dict:
        if dep.key in self.seen:
            return self.seen[dep.key]
        m = models_for(dep)
        cfg = te.tier1_config(tier2_config(dep), m)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="ctl-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(self.rows, tmp / "w.jsonl")
            te.run(tmp / "c.toml", wl, tmp / "r.jsonl", tmp / "t.jsonl")
            rec = feedback.record(tmp / "r.jsonl", dep.gpus, tmp / "t.jsonl", cfg, m["p"],
                                  dep.prefill_budget, self.slo)
        entry = {"time": time.time(), "tier": 1, "deployment": asdict(dep), "key": dep.key,
                 "workload": self.label, "rationale": rationale, "objectives": objectives(rec),
                 "bottlenecks": rec["bottlenecks"], "feedback": rec,
                 "surrogates": {k: v.key for k, v in m.items()},
                 "wall_s": time.perf_counter() - t0}
        with self.ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self.seen[dep.key] = entry
        self._archive(entry)
        return entry

    def _archive(self, entry: dict) -> None:
        o = entry["objectives"]
        if any(dominates(a["objectives"], o) for a in self.archive):
            return
        self.archive = [a for a in self.archive if not dominates(o, a["objectives"])] + [entry]


BUDGETS = (128, 512, 2048)


def neighbours(dep: Deployment) -> list[tuple[Deployment, str]]:
    """One-step moves in every dimension of the current space (fixed GPUs)."""
    out = []
    if dep.decode_instances > 1:
        out.append((replace(dep, prefill_instances=dep.prefill_instances + 1,
                            decode_instances=dep.decode_instances - 1), "neighbour: D->P"))
    if dep.prefill_instances > 1:
        out.append((replace(dep, prefill_instances=dep.prefill_instances - 1,
                            decode_instances=dep.decode_instances + 1), "neighbour: P->D"))
    # Engine constraint: heterogeneous (MLA+KDA) state handoff requires equal
    # P/D attention TP; mixed TP needs a state-resharding mechanism (Phase 4).
    other = 8 if dep.prefill_tp == 4 else 4
    out.append((replace(dep, prefill_tp=other, decode_tp=other), f"neighbour: TP {other}"))
    i = BUDGETS.index(dep.prefill_budget)
    for j in (i - 1, i + 1):
        if 0 <= j < len(BUDGETS):
            out.append((replace(dep, prefill_budget=BUDGETS[j]), f"neighbour: budget {BUDGETS[j]}"))
    return out


def rule_policy(entry: dict) -> list[tuple[Deployment, str]]:
    """Map diagnosed bottlenecks to moves at fixed GPU count."""
    dep = Deployment.of(entry["deployment"])
    labels = {b["class"] for b in entry["bottlenecks"]}
    moves = []
    if labels & {"prefill_saturated", "ttft_queueing"} and dep.prefill_budget < BUDGETS[-1]:
        bigger = BUDGETS[BUDGETS.index(dep.prefill_budget) + 1]
        moves.append((replace(dep, prefill_budget=bigger),
                      f"prefill-side bottleneck {sorted(labels)}: budget {dep.prefill_budget}->{bigger} "
                      "(fewer, larger chunks amortize weight reads and MoE dispatch)"))
    if labels & {"prefill_saturated", "ttft_queueing"} and dep.decode_instances > 1:
        moves.append((replace(dep, prefill_instances=dep.prefill_instances + 1,
                              decode_instances=dep.decode_instances - 1),
                      f"prefill-side bottleneck {sorted(labels)}: shift one instance D->P"))
    if "decode_batch_saturated" in labels and dep.prefill_instances > 1:
        moves.append((replace(dep, prefill_instances=dep.prefill_instances - 1,
                              decode_instances=dep.decode_instances + 1),
                      f"decode-side bottleneck {sorted(labels)}: shift one instance P->D"))
    return moves


def _score(entry: dict) -> float:
    o = entry["objectives"]
    return o.get("goodput_tokens_per_s_gpu") or o["tokens_per_s_gpu"]


def search(controller: Controller, start: Deployment, policy=rule_policy, budget: int = 10,
           allowed=lambda dep: True) -> list[dict]:
    """Best-first search. Candidates are (parent score, priority, move): the
    policy's diagnosis-driven moves and one coverage probe per dimension
    (neighbours) are proposed from every evaluated deployment; the candidate
    whose parent has the highest SLO goodput is evaluated next, diagnosis
    moves before probes. E0019/E0021 record why: BFS over rules alone missed
    dimensions no rule mentions (TP) and did not exploit improvements."""
    import heapq, itertools
    tie = itertools.count()
    heap = [(0.0, 0, next(tie), start, "baseline")]
    while heap and len(controller.seen) < budget:
        _, _, _, dep, why = heapq.heappop(heap)
        if dep.key in controller.seen:
            continue
        try:
            entry = controller.evaluate(dep, why)
        except KeyError as missing:     # outside validated surrogate coverage
            with controller.ledger.open("a") as f:
                f.write(json.dumps({"key": dep.key, "skipped": str(missing), "rationale": why}) + "\n")
            controller.seen[dep.key] = {"skipped": True}
            continue
        score = _score(entry)
        for priority, moves in ((0, policy(entry)),
                                (1, [(n, f"coverage probe ({r})") for n, r in neighbours(dep)])):
            for nxt, reason in moves:
                if nxt.key not in controller.seen and allowed(nxt):
                    heapq.heappush(heap, (-score, priority, next(tie), nxt, reason))
    return controller.archive


def search_quality(found: list[dict], reference: list[dict]) -> dict:
    """Compare a search's evaluated set with a reference (e.g. full grid):
    per-objective best found, and normalized hypervolume ratio (3 objectives,
    reference point = worst value of each objective in `reference`)."""
    keys = list(OBJECTIVES)
    def best(entries, k):
        vals = [e["objectives"][k] for e in entries if e["objectives"][k] is not None]
        return max(vals) if OBJECTIVES[k] > 0 else min(vals)
    per = {k: best(found, k) == best(reference, k) for k in keys}
    lo = {k: min(e["objectives"][k] for e in reference) for k in keys}
    hi = {k: max(e["objectives"][k] for e in reference) for k in keys}
    def norm(o):  # map to [0,1], larger is better
        return [((o[k] - lo[k]) if OBJECTIVES[k] > 0 else (hi[k] - o[k])) / ((hi[k] - lo[k]) or 1)
                for k in keys]
    def hv(entries, samples=20000):
        import random
        rng = random.Random(0)
        pts = [norm(e["objectives"]) for e in entries]
        hit = 0
        for _ in range(samples):
            x = [rng.random() for _ in keys]
            hit += any(all(p[i] >= x[i] for i in range(len(keys))) for p in pts)
        return hit / samples
    ref = hv(reference)
    return {"best_found": per, "hypervolume_ratio": hv(found) / ref if ref else None}
