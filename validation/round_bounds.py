"""Independent round-level differential check for the shared execution path.

The Kimi/Qwen shared paths have no isolated RoofLang replay. This module
checks one round from the worker's *exported graph* (Python bridge), without
running the Rust engine, and brackets the engine-measured round time:

1. Pricing identity: each operator's compute time equals
   sum_dtype FLOPs / preset peak (recomputed here); each demand vector equals
   the operator's declared compute/HBM/network service.
2. Lower bound: max(critical path of isolated operator times,
   max per-resource total service, max per-stream serial time).
3. Upper bound: sum of isolated operator times (fully serial execution).

Engine time outside [LB, UB] indicates a timing defect (double counting,
lost demand, missing dependency); inside says nothing about hardware accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from adapters.cost_model import preset_gpu


@dataclass
class RoundBounds:
    operations: int
    lower_s: float
    upper_s: float
    critical_path_s: float
    max_resource_load_s: float
    max_resource: str
    max_stream_serial_s: float
    pricing_errors: list = field(default_factory=list)

    def check(self, engine_s: float, rel_tol: float = 1e-6) -> list[str]:
        problems = list(self.pricing_errors)
        if engine_s < self.lower_s * (1 - rel_tol):
            problems.append(f"engine {engine_s:.9g}s < lower bound {self.lower_s:.9g}s")
        if engine_s > self.upper_s * (1 + rel_tol):
            problems.append(f"engine {engine_s:.9g}s > serial upper bound {self.upper_s:.9g}s")
        return problems


def isolated_time(work: dict) -> float:
    """Time of one operator alone: each concurrent component is limited by
    its slowest resource after its setup latency; the operator ends with its
    slowest component."""
    return max(
        (d.get("latency", 0.0) + max(d["resources"].values(), default=0.0)
         for d in work["demands"]),
        default=0.0,
    )


def _pricing_errors(ops: list[dict], hardware: str, rel_tol: float) -> list[str]:
    peaks = preset_gpu(hardware)[0].tflops
    errors = []
    for op in ops:
        details = op["work"].get("details", {})
        flops = details.get("flops_by_dtype")
        if flops is None:
            continue
        expected_us = sum(f / (peaks[d] * 1e6) for d, f in flops.items() if f)
        if abs(expected_us - details["compute_us"]) > rel_tol * max(expected_us, 1e-12):
            errors.append(f"{details['name']}: compute_us {details['compute_us']} != {expected_us}")
        compute = sum(v for d in op["work"]["demands"]
                      for r, v in d["resources"].items() if r.startswith("compute:"))
        if abs(compute - details["compute_us"] * 1e-6) > rel_tol * max(compute, 1e-15):
            errors.append(f"{details['name']}: compute demand {compute}s != {details['compute_us']}us")
    return errors


def round_bounds(ops: list[dict], hardware: str, rel_tol: float = 1e-9) -> RoundBounds:
    finish = {}
    load = defaultdict(float)
    stream = defaultdict(float)
    serial = 0.0
    for op in ops:  # exported in topological order (parents < id)
        t = isolated_time(op["work"])
        serial += t
        finish[op["id"]] = max((finish[p] for p in op["parents"]), default=0.0) + t
        for d in op["work"]["demands"]:
            for r, v in d["resources"].items():
                load[r] += v
        for s in op["work"].get("slots", []):
            stream[s] += t
    resource, max_load = max(load.items(), key=lambda kv: kv[1], default=("", 0.0))
    cp = max(finish.values(), default=0.0)
    max_stream = max(stream.values(), default=0.0)
    return RoundBounds(
        operations=len(ops), lower_s=max(cp, max_load, max_stream), upper_s=serial,
        critical_path_s=cp, max_resource_load_s=max_load, max_resource=resource,
        max_stream_serial_s=max_stream,
        pricing_errors=_pricing_errors(ops, hardware, 1e-6),
    )
