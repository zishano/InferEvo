"""Exact Tier-2 P/D configs for a deployment, via the engine's own launcher.

The launcher (inspect_rooflang.py) owns GPU allocation, KV topology and
scheduler wiring; we run it on a one-request smoke workload and keep only
its config.toml (run outputs live in a temporary directory that is always
removed). Configs are cached under experiments/_configs by deployment key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from adapters.engines import ROOT, SIMTHESIZER

# Launcher configs embed the engine tree's worker paths: one cache per tree
# (a mechanism candidate gets its own, never the stable engine's).
CACHE = ROOT / "experiments" / "_configs"
if SIMTHESIZER.resolve() != (ROOT / "engines" / "simthesizer").resolve():
    CACHE = CACHE / f"tree-{SIMTHESIZER.name}"


@dataclass(frozen=True)
class Deployment:
    """Kimi-K3 P/D layout. Per instance: GPUs = PP x (attention TP x DP) = PP x (EP x ETP).

    PP stages hold contiguous layers; each stage has TP x DP = EP x ETP GPUs.
    EP counts expert groups (896 / EP experts each); every expert is split over
    ETP consecutive GPUs. The engine's `ep` is the per-stage rank count EP x ETP.
    """
    hardware: str = "h200"
    prefill_instances: int = 1
    decode_instances: int = 1
    prefill_tp: int = 4
    decode_tp: int = 4
    prefill_ep: int = 16
    decode_ep: int = 16
    prefill_etp: int = 1
    decode_etp: int = 1
    prefill_pp: int = 1
    decode_pp: int = 1
    prefill_budget: int = 128
    decode_budget: int = 32
    prefill_max_batch: int = 32
    decode_max_batch: int = 32
    # M-MOECHUNK: expert-chunk pipelined grouped MoE (1 = off, the grouped lowering)
    prefill_moe_chunks: int = 1
    decode_moe_chunks: int = 1

    @classmethod
    def of(cls, d: dict) -> "Deployment":
        """From a ledger dict; older ledgers carry one shared `ep`."""
        d = dict(d)
        if "ep" in d:
            d["prefill_ep"] = d["decode_ep"] = d.pop("ep")
        return cls(**d)

    def stage_gpus(self, side: str) -> int:
        """GPUs per PP stage on `side` ("prefill" | "decode"): EP x ETP."""
        return getattr(self, f"{side}_ep") * getattr(self, f"{side}_etp")

    def width(self, side: str) -> int:
        """GPUs per instance on `side`: PP x EP x ETP."""
        return getattr(self, f"{side}_pp") * self.stage_gpus(side)

    @property
    def gpus(self) -> int:
        return self.prefill_instances * self.width("prefill") + self.decode_instances * self.width("decode")

    @property
    def key(self) -> str:
        pair = lambda a, b: a if a == b else f"{a}-{b}"
        ep = pair(self.prefill_ep, self.decode_ep)
        etp = "" if (self.prefill_etp, self.decode_etp) == (1, 1) else f"-etp{pair(self.prefill_etp, self.decode_etp)}"
        pp = "" if (self.prefill_pp, self.decode_pp) == (1, 1) else f"-pp{pair(self.prefill_pp, self.decode_pp)}"
        mc = (self.prefill_moe_chunks, self.decode_moe_chunks)
        mc = "" if mc == (1, 1) else f"-mc{pair(*mc)}"
        return (f"{self.hardware}-p{self.prefill_instances}x{self.prefill_tp}-"
                f"d{self.decode_instances}x{self.decode_tp}-ep{ep}{etp}{pp}-"
                f"b{self.prefill_budget}-{self.decode_budget}-m{self.prefill_max_batch}-{self.decode_max_batch}{mc}")


HEADS, EXPERTS = 96, 896      # Kimi-K3 attention heads, routed experts
MOE_CHUNKS = (1, 2, 4, 8)     # M-MOECHUNK values (spec mechanisms/specs/M-MOECHUNK.toml)


def check(dep: Deployment, context: int | None = None) -> list[str]:
    """Why `dep` cannot run (empty if legal). `context` = ISL + OSL for the KV check.

    Rules: launcher (inspect_rooflang.py) and engine (pd_disaggregation.rs)
    constraints, plus resident memory from validation.capacity.
    """
    err = []
    if dep.hardware not in ("h200", "b300"):
        err.append(f"hardware {dep.hardware}: Kimi-K3 runs on h200 or b300")
    for side in ("prefill", "decode"):
        n, tp, ep, etp, pp = (getattr(dep, f"{side}_{f}") for f in ("instances", "tp", "ep", "etp", "pp"))
        budget, batch = getattr(dep, f"{side}_budget"), getattr(dep, f"{side}_max_batch")
        gpus = ep * etp
        if n < 1:
            err.append(f"{side} instances {n}: must be >= 1")
        if pp not in range(1, 9):
            err.append(f"{side} PP{pp}: pipeline stages must be 1..8")
        if tp not in (1, 2, 4, 8) or HEADS % tp:
            err.append(f"{side} TP{tp}: attention TP must be 1, 2, 4 or 8")
        if etp not in (1, 2, 4, 8):
            err.append(f"{side} ETP{etp}: expert TP must be 1, 2, 4 or 8")
        elif not (ep >= 1 and EXPERTS % ep == 0):
            err.append(f"{side} EP{ep}: must divide {EXPERTS} experts")
        elif not (gpus & (gpus - 1) == 0 and gpus <= 128):
            err.append(f"{side} EP{ep} x ETP{etp} = {gpus} GPUs per stage: must be a power of two <= 128")
        elif gpus % tp:
            err.append(f"{side} {gpus} GPUs per stage (EP{ep} x ETP{etp}): must be a multiple of TP{tp}")
        if budget < 1 or batch < 1:
            err.append(f"{side} token budget {budget} / max batch {batch}: must be >= 1")
        if getattr(dep, f"{side}_moe_chunks") not in MOE_CHUNKS:
            err.append(f"{side} MoE chunks {getattr(dep, f'{side}_moe_chunks')}: must be one of {MOE_CHUNKS}")
    if dep.prefill_tp != dep.decode_tp:
        err.append(f"prefill TP{dep.prefill_tp} != decode TP{dep.decode_tp}: MLA+KDA state handoff "
                   "needs equal attention TP (resharding not modelled: C-STATERESHARD)")
    if err:
        return err
    from validation.capacity import Layout, evaluate
    for side in ("prefill", "decode"):
        tp, batch, ep = getattr(dep, f"{side}_tp"), getattr(dep, f"{side}_max_batch"), dep.stage_gpus(side)
        # Per-GPU expert bytes depend only on the stage GPU count: 896/EP experts x 1/ETP each.
        r = evaluate(Layout(dep.hardware, ep, tp, getattr(dep, f"{side}_pp")), contexts=(context or 1,))
        if r.status == "infeasible_weights_oom":
            err.append(f"{side} {ep} GPUs/stage TP{tp} PP{getattr(dep, side + '_pp')}: weights {r.weight_bytes_per_gpu / 1e9:.0f} GB/GPU exceed "
                       f"usable {r.usable_bytes_per_gpu / 1e9:.0f} GB {dep.hardware.upper()} HBM")
        elif r.status != "feasible":
            err.append(f"{side} {ep} GPUs/stage TP{tp} on {dep.hardware}: {r.status}")
        elif context and r.max_requests[context] < batch:
            err.append(f"{side} KV: {r.max_requests[context]} requests of {context} tokens fit per instance, "
                       f"max batch is {batch}")
    return err


# Reference Kimi operator lowering for Tier-2 (MC-5): applied to every
# instance, including configs cached before the launcher wrote it.
LOWERING = "grouped"


def lowering(chunks: int):
    """The runner's `lowering` value for one side."""
    return LOWERING if chunks == 1 else {"kind": LOWERING, "moe_chunks": chunks}


def tier2_config(dep: Deployment) -> dict:
    """Launcher config (cached per launcher arguments), lowering set per side."""
    path = CACHE / f"{replace(dep, prefill_moe_chunks=1, decode_moe_chunks=1).key}.toml"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        hw = dep.hardware
        args = ["bash", "run_kimi_k3_shared.sh",
                "--gpu-model", hw, "--prefill-gpu-model", hw, "--decode-gpu-model", hw,
                "--tp", str(dep.prefill_tp), "--prefill-tp", str(dep.prefill_tp),
                "--decode-tp", str(dep.decode_tp),
                "--prefill-ep", str(dep.stage_gpus("prefill")), "--decode-ep", str(dep.stage_gpus("decode")),
                "--prefill-pp", str(dep.prefill_pp), "--decode-pp", str(dep.decode_pp),
                "--prefill-expert-tp", str(dep.prefill_etp), "--decode-expert-tp", str(dep.decode_etp),
                "--prefill-instances", str(dep.prefill_instances),
                "--decode-instances", str(dep.decode_instances),
                "--prefill-token-budget", str(dep.prefill_budget),
                "--decode-token-budget", str(dep.decode_budget),
                "--prefill-max-batch", str(dep.prefill_max_batch),
                "--decode-max-batch", str(dep.decode_max_batch)]
        with tempfile.TemporaryDirectory(prefix="topo-") as tmp:
            out = Path(tmp) / "run"
            subprocess.run(args + ["--output-dir", str(out)], cwd=SIMTHESIZER, check=True,
                           capture_output=True)
            part = path.with_name(f"{path.name}.{os.getpid()}.part")    # atomic: concurrent searches
            shutil.copy(out / "config.toml", part)
            os.replace(part, path)
        meta = path.with_suffix(".json")
        part = meta.with_name(f"{meta.name}.{os.getpid()}.part")
        part.write_text(json.dumps(asdict(replace(dep, prefill_moe_chunks=1, decode_moe_chunks=1)), indent=1))
        os.replace(part, meta)
    cfg = tomllib.loads(path.read_text())
    for side in ("prefill", "decode"):
        for leaf in cfg["system"][f"{side}_instances"]:
            leaf["model_runner"]["lowering"] = lowering(getattr(dep, f"{side}_moe_chunks"))
    return cfg
