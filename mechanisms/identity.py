"""Default-off identity (compatibility check): with every new design variable
at its default, a candidate engine must reproduce its parent byte for byte.

Frozen cases cover both tiers and the layouts in use: Tier-2 PP1 x EP16,
Tier-2 PP2 x EP8 and Tier-1 at the MC-5 optimum. Each case is executed in a
subprocess with INFEREVO_SIMTHESIZER pointing at the tree under test, so the
tree's own launcher, bridge and binary produce the config and the result.

    ./ie.sh -m mechanisms.identity run <case> <out_dir>        (inside one tree)
    identity.compare(base_tree, candidate_tree) -> {case: {...}}
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters.engines import ROOT

CASES = {
    "t2-tp4-ep16": {"tier": 2, "dep": {}, "workload": (4, 2.0, 2048, 8, 7)},
    "t2-tp4-pp2-ep8": {"tier": 2, "dep": {"prefill_ep": 8, "decode_ep": 8, "prefill_pp": 2, "decode_pp": 2},
                       "workload": (4, 2.0, 2048, 8, 7)},
    "t1-optimum": {"tier": 1, "dep": {"prefill_instances": 4, "decode_instances": 4, "prefill_ep": 4,
                                      "prefill_pp": 4, "prefill_budget": 2048, "decode_budget": 64,
                                      "decode_max_batch": 64},
                   "workload": (200, 8.0, 8192, 1024, 7)},
}


def run_case(name: str, out: Path) -> dict:
    """Runs in the tree named by INFEREVO_SIMTHESIZER (module constants)."""
    from adapters import tier1_engine as te, workloads
    from adapters.engines import SIMTHESIZER
    from adapters.topologies import Deployment, tier2_config
    from agent.controller import models_for
    case = CASES[name]
    dep = Deployment(hardware="h200", **case["dep"])
    out.mkdir(parents=True, exist_ok=True)
    cfg = tier2_config(dep)
    if case["tier"] == 1:
        cfg = te.tier1_config(cfg, models_for(dep))
    te.write_config(cfg, out / "config.toml")
    wl = workloads.write(workloads.fixed(*case["workload"][:4], seed=case["workload"][4]), out / "w.jsonl")
    subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(out / "config.toml"),
                    "--workload", str(wl), "--output-jsonl", str(out / "r.jsonl")], check=True, capture_output=True)
    return {"result_sha256": hashlib.sha256((out / "r.jsonl").read_bytes()).hexdigest(),
            "config": (out / "config.toml").read_text()
                      .replace(str(ROOT / "engines" / "simthesizer" / ".venv"), "<VENV>")
                      .replace(str(SIMTHESIZER / ".venv"), "<VENV>").replace(str(SIMTHESIZER), "<TREE>")}


def _in_tree(tree: Path, name: str, out: Path) -> dict:
    env = dict(os.environ, INFEREVO_SIMTHESIZER=str(tree),
               UV_PROJECT_ENVIRONMENT=str(ROOT / "engines" / "simthesizer" / ".venv"))
    r = subprocess.run([str(ROOT / "ie.sh"), "-m", "mechanisms.identity", "run", name, str(out)],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        return {"error": (r.stderr or r.stdout).strip().splitlines()[-5:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def compare(base: Path, candidate: Path, cases=None) -> dict:
    out = {}
    names = list(cases or CASES)
    with tempfile.TemporaryDirectory(prefix="identity-") as tmp, ThreadPoolExecutor(2 * len(names)) as pool:
        runs = {(n, side): pool.submit(_in_tree, tree, n, Path(tmp) / side / n)
                for n in names for side, tree in (("base", base), ("cand", candidate))}
        for name in names:
            a, b = runs[(name, "base")].result(), runs[(name, "cand")].result()
            if "error" in a or "error" in b:
                out[name] = {"identical": False, "base_error": a.get("error"), "candidate_error": b.get("error")}
                continue
            out[name] = {"identical": a["result_sha256"] == b["result_sha256"],
                         "config_identical": a["config"] == b["config"],
                         "base_sha256": a["result_sha256"], "candidate_sha256": b["result_sha256"]}
    return out


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "run":
        r = run_case(sys.argv[2], Path(sys.argv[3]))
        print(json.dumps(r))
    else:
        raise SystemExit(__doc__)
