"""Cost-model identity, vendored-copy drift, precision-path sensitivity.

The executable cost model of the integrated simulator is Simthesizer's
vendored RoofLang subset plus the bridge lowering (cost conventions such as
top-k pricing live there). Its content hash is the `cost_model_version`
bound to every result and to Tier-1 cache keys (DECISIONS D3/D6).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from adapters.engines import ROOFLANG, SIMTHESIZER, enable_bridge_imports

VENDORED = SIMTHESIZER / "scripts" / "rooflang"
BRIDGE = SIMTHESIZER / "scripts" / "rooflang_bridge"
WORKER = SIMTHESIZER / "scripts" / "rooflang_worker.py"


def _py_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def cost_model_files() -> list[Path]:
    return _py_files(VENDORED) + _py_files(BRIDGE) + [WORKER]


def cost_model_version() -> str:
    h = hashlib.sha256()
    for path in cost_model_files():
        h.update(str(path.relative_to(SIMTHESIZER)).encode() + b"\0")
        h.update(path.read_bytes() + b"\0")
    return h.hexdigest()


def vendored_drift() -> list[dict]:
    """Vendored RoofLang files that differ from or are missing upstream."""
    out = []
    for path in _py_files(VENDORED):
        rel = path.relative_to(VENDORED)
        upstream = ROOFLANG / rel
        if not upstream.exists():
            out.append({"file": str(rel), "kind": "missing_upstream"})
        elif upstream.read_bytes() != path.read_bytes():
            out.append({"file": str(rel), "kind": "content_differs"})
    return out


def preset_gpu(preset: str):
    """First GPU of the RoofLang preset vendored by Simthesizer."""
    enable_bridge_imports()
    if preset == "h200":
        from rooflang.programs.presets.h200 import H200Cluster as Cluster
    elif preset == "b300":
        from rooflang.programs.presets.b300 import B300Cluster as Cluster
    else:
        raise ValueError(preset)
    hw = Cluster(n_nodes=1)
    gpu = sorted((n for n in hw.nodes if n.kind == "gpu"), key=lambda n: n.name)[0]
    hbm = hw.find_local_memory(gpu)
    fabric = hw.find_fabric(gpu, hbm)
    return gpu, hbm, fabric


def roofline_breakdown(hardware: str, batch: int, tokens: int,
                       context: int | None, fp4_rate: float | None = None) -> dict:
    """Serial single-device roofline over the logical Kimi graph.

    Sum over operators of max(compute, HBM) time with the preset's peaks,
    ignoring parallelism, communication and overlap between operators. Only
    valid for *ratios* (e.g. sensitivity to a precision path), never as a
    latency prediction. `fp4_rate` overrides the preset FP4 TFLOP/s to model
    an emulated path (e.g. H200 W4A16 at the BF16 rate).
    """
    enable_bridge_imports()
    from rooflang_bridge import kimi_k3 as K
    from rooflang.runtime.simulator import Simulator

    gpu, _, fabric = preset_gpu(hardware)
    peaks = dict(gpu.tflops)
    if fp4_rate is not None:
        peaks["fp4"] = fp4_rate
    bw = fabric.src_to_dst_bandwidth_gbs * 1e3  # bytes per µs
    nodes, _ = K.compact_graph(batch, tokens, context)
    total = compute_bound = 0.0
    by_dtype_compute = {}
    for node in nodes:
        kernel, m = node["kernel"], node["multiplicity"]
        flops = kernel.flops_by_dtype or {Simulator._infer_dtype(kernel): kernel.flops}
        ct = 0.0
        for dtype, f in flops.items():
            if f:
                rate = peaks.get(dtype, 0.0)
                if rate <= 0:
                    raise ValueError(f"{hardware} has no {dtype} rate")
                t = float(f) * m / (rate * 1e6)
                ct += t
                by_dtype_compute[dtype] = by_dtype_compute.get(dtype, 0.0) + t
        mt = float(kernel.transferred_bytes) * m / bw
        total += max(ct, mt)
        if ct > mt:
            compute_bound += ct
    return {"serial_roofline_us": total, "compute_bound_us": compute_bound,
            "compute_us_by_dtype": by_dtype_compute}
