"""Mechanism specification (spec 8.2): load, validate, plan.

A spec is `mechanisms/specs/<ID>.toml`, written before implementation:

    id = "M-OVERLAP"                 # unique
    name = "..."
    parent = "simthesizer@<commit>"  # base version (or a parent mechanism id)
    catalog = "C-OVERLAP"            # catalog candidate it realises
    bottleneck = "..."               # what it addresses, with evidence refs
    evidence = ["E0039", ...]
    hypothesis = "..."
    expected_impact = "..."
    repos = ["simthesizer"]
    [components]
    added = []                       # new files (registry ids that must not exist yet)
    modified = ["sim:..."]           # must exist
    reused = ["sim:..."]             # must exist; not edited
    [interface]                      # state / interface / config changes
    config = ["model_runner.moe_overlap: bool = false"]
    state = []
    [design_variables]               # new searchable knobs; default must be the off state
    moe_overlap = { values = [false, true], default = false }
    resources = "..."                # extra memory / links, if any
    semantics = ["work conserved", ...]
    failure_modes = ["..."]
    [validation]
    default_off_identity = true      # mechanism off => results byte-identical to parent
    checks = [["./ie.sh", "-m", "unittest", "tests.test_x"]]     # extra commands
    [evaluation]
    script = "experiments/EXXXX/run.py"    # A-E ablation (D24), prints a JSON verdict
    rollback = "..."

`plan()` checks the spec against the component registry and returns the
extension order (dependencies first), the predicted impact closure and the
regression tests that closure selects.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from mechanisms.registry import Registry

SPECS = Path(__file__).with_name("specs")
REQUIRED = ("id", "name", "parent", "bottleneck", "hypothesis", "expected_impact", "repos",
            "components", "semantics", "failure_modes", "validation", "evaluation", "rollback")


def load(path_or_id) -> dict:
    p = Path(path_or_id)
    if not p.suffix:
        p = SPECS / f"{path_or_id}.toml"
    spec = tomllib.loads(p.read_text())
    spec["_path"] = str(p)
    return spec


def validate(spec: dict, reg: Registry | None = None) -> list[str]:
    reg = reg or Registry()
    errors = [f"missing field `{k}`" for k in REQUIRED if k not in spec]
    comps = spec.get("components", {})
    for cid in comps.get("modified", []) + comps.get("reused", []):
        if cid not in reg.c:
            errors.append(f"unknown component {cid}")
    for cid in comps.get("added", []):
        if cid in reg.c:
            errors.append(f"`added` component already exists: {cid}")
    if not comps.get("modified") and not comps.get("added"):
        errors.append("a mechanism must add or modify at least one component")
    for name, var in spec.get("design_variables", {}).items():
        if "default" not in var or "values" not in var:
            errors.append(f"design variable {name}: needs `values` and `default`")
        elif var["default"] not in var["values"]:
            errors.append(f"design variable {name}: default not among values")
    if spec.get("validation", {}).get("default_off_identity") and not spec.get("design_variables"):
        errors.append("default_off_identity needs a design variable whose default is the off state")
    script = spec.get("evaluation", {}).get("script")
    if script and not (Path(__file__).resolve().parents[1] / script).exists():
        errors.append(f"evaluation script {script} does not exist")
    return errors


def plan(spec: dict, reg: Registry | None = None) -> dict:
    reg = reg or Registry()
    comps = spec["components"]
    touched = set(comps.get("modified", []))
    order = reg.topological(touched | set(comps.get("reused", [])))
    impact = reg.closure(touched)
    return {"extension_order": [c for c in order if c in touched],
            "context": [c for c in order if c not in touched],
            "impact": sorted(impact), "tests": [t.id for t in reg.tests_for(touched)]}
