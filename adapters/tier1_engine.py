"""Derive a Tier-1 engine config from a Tier-2 (shared Rooflang) Kimi config.

Only execution changes: every instance's `rooflang` runner becomes a native
`step_model` runner with fitted coefficients for its stage, models switch to
the scalar Kimi description, and the system runs non-shared. Schedulers,
P/D pools, KV topology, hardware allocation and workload stay identical, so
Tier-1 vs Tier-2 differences isolate the cost surrogate (plus the documented
scalar-state approximations: whole-request state reservation and one MLA
replica per P->D handoff). Scheduler limits are converted from per-instance
(shared) to the legacy per-rank convention (divided by EP).

PP: the fitted model predicts a whole round (all stages, boundary sends
included); the step runner splits it by each stage's layer share and
overlaps up to PP rounds. The scalar P->D handoff spreads over the
stage-major group (per-layer stage ownership is a Tier-2 detail).
"""

from __future__ import annotations

import copy
import subprocess
import tomllib
from pathlib import Path

from adapters.engines import SIMTHESIZER, enable_bridge_imports
from adapters.tier1 import Tier1Model

SCALAR_MODULE = "rooflang_bridge.kimi_k3_scalar"


def _scalar_model(model: dict) -> dict:
    model = copy.deepcopy(model)
    cmd = model["command"]
    cmd[cmd.index("--module") + 1] = SCALAR_MODULE
    return model


def tier1_config(tier2: dict, models: dict[str, Tier1Model]) -> dict:
    """models: {"p": prefill model, "d": decode model}."""
    cfg = copy.deepcopy(tier2)
    system = cfg["system"]
    system["shared_execution"] = False
    for key in ("allow_gpu_sharing", "kv_handoff"):
        system.pop(key, None)
    system["model"] = _scalar_model(system["model"])
    for key in ("prefill_layer_stages", "decode_layer_stages"):
        system.get("kv_topology", {}).pop(key, None)
    enable_bridge_imports()
    from rooflang_bridge.kimi_k3_shared import layer_stages
    for side, stage in (("prefill_instances", "p"), ("decode_instances", "d")):
        for leaf in system[side]:
            runner = leaf["model_runner"]
            m = models[stage]
            pp = runner.get("pp", 1)
            if (m.key["tp"], m.key["ep"], m.key.get("pp", 1)) != (runner["tp"], runner["ep"], pp):
                raise ValueError(f"{stage} surrogate layout {m.key} != runner TP/EP/PP")
            lowering = runner.get("lowering")
            chunks = lowering.get("moe_chunks", 1) if isinstance(lowering, dict) else 1
            if m.key.get("moe_chunks", 1) != chunks:
                raise ValueError(f"{stage} surrogate fitted at MoE chunks {m.key.get('moe_chunks', 1)}, runner uses {chunks}")
            stages = layer_stages(pp)
            leaf["model_runner"] = {
                "kind": "step_model", "tp": runner["tp"], "ep": runner["ep"],
                "groups": m.key["groups"], "coefficients": list(m.coef),
                **({"instance_id": runner["instance_id"]} if "instance_id" in runner else {}),
                **({"stage_fractions": [stages.count(s) / len(stages) for s in range(pp)]} if pp > 1 else {}),
            }
            leaf["model"] = _scalar_model(leaf["model"])
            # The legacy scheduler treats batch/token limits as per-rank and
            # multiplies them by EP; shared Kimi applies them per instance.
            sched, ep = leaf["scheduler"], runner["ep"]
            for key in ("max_batch", "max_num_batched_tokens"):
                if sched[key] % ep:
                    raise ValueError(f"{key}={sched[key]} not divisible by EP={ep}")
                sched[key] //= ep
    return cfg


def write_config(cfg: dict, path: Path) -> None:
    enable_bridge_imports()
    from rooflang_bridge.configuration import toml_document
    path.write_text(toml_document(cfg))


def run(config: Path, workload: Path, output: Path, trace: Path | None = None) -> subprocess.CompletedProcess:
    """Run the engine; `trace` adds the (small, Tier-1) graph trace for feedback."""
    binary = SIMTHESIZER / "target" / "release" / "simthesizer"
    subprocess.run(["cargo", "build", "--release", "--locked", "-q"], cwd=SIMTHESIZER,
                   check=True, capture_output=True)
    extra = ["--output-graph-trace", str(trace)] if trace else []
    return subprocess.run([str(binary), "--config", str(config), "--workload", str(workload),
                           "--output-jsonl", str(output), *extra], check=True,
                          capture_output=True, text=True)


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text())
