"""Mechanism lifecycle (Phase 4): spec -> candidate -> checks -> A-E evaluation
-> replay -> promote / reject. Only this controller writes states and
receipts; candidates never touch the evaluator (D9).

States (ARCHITECTURE §3): DRAFT -> CHECKING -> EVALUATING -> REPLAYING ->
VERIFIED | EXPLORATORY | FAILED; VERIFIED -> PROMOTED. Every transition is
one line in mechanisms/lineage.jsonl (parent, commits, receipts), so the
mechanism genealogy and failure reasons are kept (spec 4.4.8).

    ./ie.sh -m mechanisms.manage plan <ID>       validate spec, extension order, predicted impact/tests
    ./ie.sh -m mechanisms.manage new <ID>        candidate worktrees (inferevo/cand/<id>)
    ./ie.sh -m mechanisms.manage check <ID>      declared-vs-actual diff, build, selected regression
                                                 tests, default-off identity, spec checks
    ./ie.sh -m mechanisms.manage evaluate <ID>   A-E ablation script against the candidate tree
    ./ie.sh -m mechanisms.manage replay <ID>     independent re-run of the script's replay case
    ./ie.sh -m mechanisms.manage promote <ID>    fast-forward the engine branch (VERIFIED only)
    ./ie.sh -m mechanisms.manage reject <ID> --reason ...
    ./ie.sh -m mechanisms.manage status [ID]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters.engines import ENGINES, ROOT, repo_identity
from mechanisms import identity, isolation, spec as specs
from mechanisms.registry import Registry, changed_files

LINEAGE = ROOT / "mechanisms" / "lineage.jsonl"
RECEIPTS = ROOT / "mechanisms" / "receipts"
PREFIX = {"simthesizer": "sim", "rooflang": "rl"}
# The frozen evaluator: hashed into every receipt; a candidate cannot change it.
EVALUATOR = ["adapters/steady.py", "adapters/workloads.py", "agent/search.py", "adapters/topologies.py",
             "mechanisms/identity.py", "mechanisms/manage.py"]
VENV = ENGINES / "simthesizer" / ".venv"


def _events(mid=None):
    if not LINEAGE.exists():
        return []
    rows = [json.loads(l) for l in LINEAGE.read_text().splitlines() if l.strip()]
    return [r for r in rows if mid is None or r["id"] == mid]


def state(mid):
    ev = _events(mid)
    return ev[-1]["state"] if ev else None


def _record(mid, new_state, **details):
    LINEAGE.parent.mkdir(parents=True, exist_ok=True)
    row = {"id": mid, "state": new_state, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **details}
    with LINEAGE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(mid, step, body):
    d = RECEIPTS / mid
    d.mkdir(parents=True, exist_ok=True)
    body = {"id": mid, "step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluator": {f: _sha(ROOT / f) for f in EVALUATOR}, **body}
    p = d / f"{step}.json"
    p.write_text(json.dumps(body, indent=1))
    return str(p.relative_to(ROOT))


def candidates(mid):
    """{repo: record} of the candidate worktrees created by `new`."""
    ev = [e for e in _events(mid) if e["state"] == "DRAFT"]
    if not ev:
        raise SystemExit(f"{mid}: no candidate (run `new` first)")
    return ev[-1]["candidates"]


def _tree(rec):
    return ROOT / rec["path"]


def _env(trees):
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(VENV))
    if "simthesizer" in trees:
        env["INFEREVO_SIMTHESIZER"] = str(trees["simthesizer"])
    if "rooflang" in trees:
        env["INFEREVO_ROOFLANG"] = str(trees["rooflang"])
    return env


def _run(cmd, cwd, env, log):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
    log.append({"cmd": cmd, "cwd": str(cwd), "rc": r.returncode, "wall_s": round(time.perf_counter() - t0, 1),
                "tail": tail if r.returncode else tail[-2:]})
    return r


# ---------------------------------------------------------------- commands
def cmd_plan(mid):
    s = specs.load(mid)
    reg = Registry()
    errors = specs.validate(s, reg)
    out = {"id": s.get("id"), "errors": errors}
    if not errors:
        out.update(specs.plan(s, reg))
    return out


def cmd_new(mid):
    s = specs.load(mid)
    errors = specs.validate(s)
    if errors:
        raise SystemExit("spec invalid:\n  " + "\n  ".join(errors))
    if state(mid) not in (None, "FAILED"):
        raise SystemExit(f"{mid} is {state(mid)}")
    name = mid.lower() + ("" if state(mid) is None else f"-r{sum(e['state'] == 'DRAFT' for e in _events(mid))}")
    cands = {repo: isolation.candidate(repo, name) for repo in s["repos"]}
    return _record(mid, "DRAFT", parent=s["parent"], spec=s["_path"], spec_sha256=_sha(Path(s["_path"])),
                   candidates=cands)


def cmd_check(mid):
    if state(mid) not in ("DRAFT", "CHECKING", "FAILED"):
        raise SystemExit(f"{mid} is {state(mid)}; check runs on DRAFT candidates")
    s, cands = specs.load(mid), candidates(mid)
    trees = {repo: _tree(rec) for repo, rec in cands.items()}
    _record(mid, "CHECKING")
    log, problems = [], []
    reg = Registry({PREFIX[r]: t for r, t in trees.items()})
    declared = set(s["components"].get("modified", [])) | set(s["components"].get("added", []))
    changed = set()
    for repo, rec in cands.items():
        ident = repo_identity(repo, trees[repo])
        if ident.dirty:
            problems.append(f"{repo} candidate has uncommitted changes")
        known, unknown = reg.from_files(repo, changed_files(trees[repo], rec["parent_commit"], "HEAD"))
        changed |= known
        problems += [f"{repo}: changed file outside the registry: {u}" for u in unknown]
    undeclared = sorted(c for c in changed - declared if reg.c[c].kind != "test")
    problems += [f"undeclared modification: {c}" for c in undeclared]
    unchanged = sorted(declared - changed)
    if "simthesizer" in trees:
        for profile in (["--release"], []):          # Python tests run target/{release,debug}
            if _run(["cargo", "build", "--locked", *profile], trees["simthesizer"], _env(trees), log).returncode:
                problems.append(f"cargo build {' '.join(profile) or '--debug'} failed")
    tests = reg.tests_for(changed) if not problems else []
    sim = trees.get("simthesizer", ENGINES / "simthesizer")
    cwd = {"sim": sim, "sim-tests": sim / "tests", "ie": ROOT}

    def test(t):
        cmd = [str(VENV / "bin/python") if c == ".venv/bin/python" else c for c in t.test["cmd"]]
        env = dict(_env(trees), **t.test.get("env", {}))
        if t.test["cwd"].startswith("sim"):
            env.pop("PYTHONPATH", None)               # InferEvo's own `tests` package would shadow
        return t, _run(cmd, cwd.get(t.test["cwd"], t.test["cwd"]), env, log)
    rust = [t for t in tests if t.test["kind"].startswith("rust")]
    results = [test(t) for t in rust]                 # cargo serializes on the target dir anyway
    with ThreadPoolExecutor(8) as pool:
        results += list(pool.map(test, [t for t in tests if t not in rust]))
    problems += [f"regression test failed: {t.id}" for t, r in results if r.returncode]
    ident = None
    if not problems and s["validation"].get("default_off_identity"):
        ident = identity.compare(ENGINES / "simthesizer", trees["simthesizer"])
        problems += [f"default-off identity broken: {k}" for k, v in ident.items() if not v["identical"]]
    for c in s["validation"].get("checks", []) if not problems else []:
        if _run(c, ROOT, _env(trees), log).returncode:
            problems.append(f"spec check failed: {' '.join(c)}")
    receipt = _receipt(mid, "check", {
        "commits": {repo: repo_identity(repo, t).commit for repo, t in trees.items()},
        "changed": sorted(changed), "undeclared": undeclared, "declared_unchanged": unchanged,
        "impact": sorted(reg.closure(changed)), "tests": [t.id for t in tests],
        "identity": ident, "log": log, "problems": problems})
    return _record(mid, "FAILED" if problems else "EVALUATING", receipt=receipt,
                   **({"reason": problems} if problems else {}))


def _script(mid, extra):
    s, cands = specs.load(mid), candidates(mid)
    trees = {repo: _tree(rec) for repo, rec in cands.items()}
    log = []
    r = _run(["./ie.sh", s["evaluation"]["script"], *extra], ROOT, dict(_env(trees), INFEREVO_MECHANISM=mid), log)
    try:
        verdict = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        verdict = {"verdict": "FAILED", "reason": "evaluation script printed no JSON verdict"}
    if r.returncode:
        verdict = {"verdict": "FAILED", "reason": f"evaluation script exit {r.returncode}"}
    return s, trees, log, verdict


def _checked(mid):
    """A passed check receipt for the candidates' current commits."""
    p = RECEIPTS / mid / "check.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    now = {repo: repo_identity(repo, _tree(rec)).commit for repo, rec in candidates(mid).items()}
    return not r["problems"] and r["commits"] == now


def cmd_evaluate(mid):
    last = _events(mid)[-1] if _events(mid) else {}
    retry = last.get("state") == "FAILED" and str(last.get("receipt", "")).endswith("evaluate.json")
    if not (state(mid) == "EVALUATING" or retry) or not _checked(mid):
        raise SystemExit(f"{mid} is {state(mid)}; evaluate needs a passed check of the current commits")
    s, trees, log, verdict = _script(mid, [])
    receipt = _receipt(mid, "evaluate", {
        "commits": {repo: repo_identity(repo, t).commit for repo, t in trees.items()},
        "script": s["evaluation"]["script"], "script_sha256": _sha(ROOT / s["evaluation"]["script"]),
        "verdict": verdict, "log": log})
    nxt = {"VERIFIED": "REPLAYING", "EXPLORATORY": "EXPLORATORY"}.get(verdict.get("verdict"), "FAILED")
    return _record(mid, nxt, receipt=receipt, verdict=verdict.get("verdict"),
                   **({"reason": verdict.get("reason")} if nxt == "FAILED" else {}))


def cmd_replay(mid):
    if state(mid) != "REPLAYING":
        raise SystemExit(f"{mid} is {state(mid)}; replay follows a VERIFIED evaluation")
    s, trees, log, verdict = _script(mid, ["--replay"])
    ok = verdict.get("verdict") == "REPRODUCED"
    receipt = _receipt(mid, "replay", {"commits": {repo: repo_identity(repo, t).commit for repo, t in trees.items()},
                                       "verdict": verdict, "log": log})
    return _record(mid, "VERIFIED" if ok else "FAILED", receipt=receipt,
                   **({} if ok else {"reason": verdict.get("reason", "replay differs")}))


def cmd_promote(mid):
    if state(mid) != "VERIFIED":
        raise SystemExit(f"{mid} is {state(mid)}; only VERIFIED candidates are promoted")
    merged = {}
    for repo, rec in candidates(mid).items():
        engine = isolation.ENGINE[repo]
        head = repo_identity(repo, engine)
        if head.dirty:
            raise SystemExit(f"{engine} has uncommitted changes")
        if head.commit != rec["parent_commit"]:
            raise SystemExit(f"{repo}: engine moved since the candidate was created "
                             f"({rec['parent_commit'][:12]} -> {head.commit[:12]}); re-run on the new base")
        subprocess.run(["git", "-C", str(engine), "merge", "--ff-only", rec["branch"]], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(engine), "worktree", "remove", str(_tree(rec))], check=True, capture_output=True)
        merged[repo] = repo_identity(repo, engine).commit
    return _record(mid, "PROMOTED", engine_commits=merged)


def cmd_reject(mid, reason):
    for repo, rec in candidates(mid).items():
        if _tree(rec).exists():
            subprocess.run(["git", "-C", str(isolation.ENGINE[repo]), "worktree", "remove", "--force", str(_tree(rec))],
                           check=True, capture_output=True)
    return _record(mid, "FAILED", reason=reason, branch_kept=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["plan", "new", "check", "evaluate", "replay", "promote", "reject", "status"])
    p.add_argument("id", nargs="?")
    p.add_argument("--reason", default="")
    a = p.parse_args()
    if a.cmd == "status":
        out = _events(a.id)
    elif a.cmd == "reject":
        out = cmd_reject(a.id, a.reason)
    else:
        out = globals()[f"cmd_{a.cmd}"](a.id)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
