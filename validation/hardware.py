"""Sourced H200 / B300 per-GPU specifications and precision capabilities.

Single authoritative home for hardware facts used by InferEvo checks. Each
value carries provenance; `status` distinguishes official specifications,
third-party figures, derivations and open assumptions (RESEARCH_SPEC 10.5).

RoofLang presets remain the executable cost model. `audit_preset()` compares
them with these facts so deviations become explicit cost-model change
requests instead of silent edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adapters.cost_model import preset_gpu

OFFICIAL = "official_spec"
THIRD_PARTY = "third_party_figure"
DERIVED = "derived"
ASSUMPTION = "unverified_assumption"
MEASURED = "public_measurement"


@dataclass(frozen=True)
class Fact:
    value: float
    unit: str
    status: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class GpuSpec:
    name: str
    architecture: str
    facts: dict
    # Tensor Core datatypes with a native MMA path. Formats absent here must be
    # executed by an explicit emulation path (dequantize, then compute in a
    # native type) and may not be priced at a native rate.
    native_mma: frozenset
    notes: tuple = field(default_factory=tuple)
    # Non-native storage dtype -> native dtype whose rate bounds its GEMMs
    # (e.g. MXFP4 weights dequantized to BF16 on Hopper: W4A16).
    emulation: dict = field(default_factory=dict)


H200_URL = "https://www.nvidia.com/en-us/data-center/h200/"
HGX_URL = "https://www.nvidia.com/en-us/data-center/hgx/"
SEMI_B300 = "https://inferencex.semianalysis.com/chips/b300"
BU_DATASHEET = ("NVIDIA Blackwell Ultra datasheet 4169750 (Oct 2025), table "
                "'Individual Blackwell Ultra GPU Specifications', HGX B300 column; "
                "PDF sha256 ccdffc064731cc160016f4c7a77ecfa37983038bcaf60e39190ba4175b39c7c8")
INFSIM = ("inference-sim e7597a8 engines/sglang/lower/testdata/{h200,b300}.yaml, "
          "citing CUDA total_memory reports")
TOMS_B300 = ("https://www.tomshardware.com/pc-components/gpus/"
             "nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-"
             "with-288gb-hbm3e-and-15-pflops-dense-fp4")

H200 = GpuSpec(
    name="h200_sxm",
    architecture="hopper",
    facts={
        # Datasheet "141GB" reads as GiB; the driver reports 150,109,880,320 B
        # (vLLM allocates from this total). Decimal GB, as RoofLang uses.
        "hbm_capacity_gb": Fact(150.10988032, "GB", MEASURED, INFSIM,
                                "datasheet label 141GB; CM-6"),
        "hbm_bandwidth_gbs": Fact(4800.0, "GB/s", OFFICIAL, H200_URL),
        # Datasheet lists sparse peaks; dense = sparse / 2.
        "bf16_dense_tflops": Fact(989.5, "TFLOP/s", DERIVED, H200_URL,
                                  "1979 sparse / 2"),
        "fp8_dense_tflops": Fact(1979.0, "TFLOP/s", DERIVED, H200_URL,
                                 "3958 sparse / 2"),
        "nvlink_per_direction_gbs": Fact(450.0, "GB/s", DERIVED, H200_URL,
                                         "900 GB/s bidirectional / 2"),
    },
    native_mma=frozenset({"fp32", "tf32", "bf16", "fp16", "fp8", "int8"}),
    notes=("Hopper has no FP4/MXFP4 Tensor Core path; MXFP4 weights must be "
           "dequantized (W4A16 or W4A8) before MMA.",),
    emulation={"fp4": "bf16"},
)

B300 = GpuSpec(
    name="b300_sxm",
    architecture="blackwell_ultra",
    facts={
        "hbm_capacity_gb": Fact(287.42877184, "GB", MEASURED, INFSIM,
                                "datasheet label 270 GB HBM3E; CM-6"),
        "hbm_bandwidth_gbs": Fact(7700.0, "GB/s", OFFICIAL, BU_DATASHEET,
                                  "8 TB/s is the GB300 figure; CM-6 supersedes CM-3"),
        "fp4_dense_tflops": Fact(14000.0, "TFLOP/s", OFFICIAL, BU_DATASHEET,
                                 "18 sparse | 14 dense PFLOPS"),
        "fp8_dense_tflops": Fact(4500.0, "TFLOP/s", DERIVED, BU_DATASHEET,
                                 "9 PFLOPS sparse / 2"),
        "bf16_dense_tflops": Fact(2250.0, "TFLOP/s", DERIVED, BU_DATASHEET,
                                  "4.5 PFLOPS sparse / 2"),
        "nvlink_per_direction_gbs": Fact(900.0, "GB/s", DERIVED, BU_DATASHEET,
                                         "NVLink 5: 1.8 TB/s bidirectional / 2"),
    },
    native_mma=frozenset({"fp32", "tf32", "bf16", "fp16", "fp8", "fp6",
                          "fp4", "mxfp4", "nvfp4", "mxfp8"}),
    notes=(TOMS_B300,),
)

SPECS = {"h200": H200, "b300": B300}


def audit_preset(preset: str, rel_tol: float = 0.02) -> list[dict]:
    """Findings where a RoofLang preset deviates from sourced facts."""
    spec = SPECS[preset]
    gpu, hbm, fabric = preset_gpu(preset)
    findings = []

    def compare(key, actual):
        fact = spec.facts.get(key)
        if fact is None:
            return
        if abs(actual - fact.value) > rel_tol * fact.value:
            findings.append({
                "kind": "value_deviation", "field": key,
                "preset": actual, "sourced": fact.value,
                "status": fact.status, "source": fact.source,
            })

    compare("hbm_capacity_gb", hbm.capacity_gb)
    compare("hbm_bandwidth_gbs", fabric.src_to_dst_bandwidth_gbs)
    compare("bf16_dense_tflops", gpu.tflops.get("bf16", 0.0))
    compare("fp8_dense_tflops", gpu.tflops.get("fp8", 0.0))
    compare("fp4_dense_tflops", gpu.tflops.get("fp4", 0.0))
    for dtype, rate in sorted(gpu.tflops.items()):
        if rate <= 0 or dtype in spec.native_mma:
            continue
        path = spec.emulation.get(dtype)
        if path is None or rate > gpu.tflops.get(path, 0.0):
            findings.append({
                "kind": "non_native_precision_priced_natively",
                "field": dtype, "preset": rate,
                "detail": f"{spec.architecture} has no native {dtype} MMA; "
                          f"rate exceeds emulation path {path}",
            })
    return findings
