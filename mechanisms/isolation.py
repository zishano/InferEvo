"""Candidate isolation and version records (DECISIONS D8).

baseline: snapshot a *user* checkout (HEAD + uncommitted diff [+ untracked])
          as branch `inferevo/<name>` in a new worktree. Used to create the
          engines/ trees; the user's working tree is never modified.
candidate: branch `inferevo/cand/<name>` from the active engine HEAD into
          `mechanisms/candidates/<repo>-<name>`; the stable engine is never
          overwritten by an unvalidated candidate.

Every operation writes a record to `mechanisms/versions/`.

    python -m mechanisms.isolation baseline simthesizer phase1-base engines/simthesizer --untracked
    python -m mechanisms.isolation candidate simthesizer my-mechanism
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from adapters.engines import (
    ROOFLANG, ROOT, SIMTHESIZER, USER_ROOFLANG, USER_SIMTHESIZER, repo_identity,
)

USER = {"simthesizer": USER_SIMTHESIZER, "rooflang": USER_ROOFLANG}
ENGINE = {"simthesizer": SIMTHESIZER, "rooflang": ROOFLANG}
CANDIDATES = ROOT / "mechanisms" / "candidates"
VERSIONS = ROOT / "mechanisms" / "versions"


def _git(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, **kw)


def _commit(dest: Path, message: str) -> str:
    _git(dest, "add", "-A")
    _git(dest, "-c", "user.name=inferevo", "-c", "user.email=inferevo@localhost",
         "commit", "-q", "--allow-empty", "-m", message)
    return _git(dest, "rev-parse", "HEAD", text=True).stdout.strip()


def _record(name: str, record: dict) -> dict:
    VERSIONS.mkdir(parents=True, exist_ok=True)
    (VERSIONS / f"{name}.json").write_text(json.dumps(record, indent=1))
    return record


def baseline(repo_name: str, name: str, dest: Path, untracked: bool) -> dict:
    repo = USER[repo_name]
    ident = repo_identity(repo_name, repo)
    branch = f"inferevo/{name}"
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    _git(repo, "worktree", "add", "-b", branch, str(dest), ident.commit)
    diff = _git(repo, "diff", "HEAD", "--binary").stdout
    if diff:
        subprocess.run(["git", "-C", str(dest), "apply", "--whitespace=nowarn"],
                       input=diff, check=True)
    files = _git(repo, "ls-files", "--others", "--exclude-standard", "-z").stdout
    names = [n for n in files.decode().split("\0") if n]
    copied = []
    if untracked:
        for rel in names:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / rel, dest / rel)
            copied.append(rel)
    head = _commit(dest, f"inferevo: baseline snapshot of {ident.commit[:12]}"
                         f" + working tree {ident.dirty_digest[:12] or 'clean'}"
                         f"\n\nuntracked copied: {len(copied)} / {len(names)}")
    return _record(f"{repo_name}-{name}", {
        "kind": "baseline", "repo": repo_name, "branch": branch,
        "path": str(dest.relative_to(ROOT)), "snapshot_commit": head,
        "source": ident.__dict__, "untracked_copied": len(copied),
        "untracked_total": len(names)})


def candidate(repo_name: str, name: str) -> dict:
    engine = ENGINE[repo_name]
    parent = repo_identity(repo_name, engine)
    if parent.dirty:
        raise SystemExit(f"{engine} has uncommitted changes; commit them first")
    dest = CANDIDATES / f"{repo_name}-{name}"
    branch = f"inferevo/cand/{name}"
    _git(engine, "worktree", "add", "-b", branch, str(dest), parent.commit)
    return _record(f"cand-{repo_name}-{name}", {
        "kind": "candidate", "repo": repo_name, "branch": branch,
        "path": str(dest.relative_to(ROOT)), "parent_branch": parent.branch,
        "parent_commit": parent.commit})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("repo", choices=USER)
    b.add_argument("name")
    b.add_argument("dest", type=Path)
    b.add_argument("--untracked", action="store_true")
    c = sub.add_parser("candidate")
    c.add_argument("repo", choices=ENGINE)
    c.add_argument("name")
    a = p.parse_args()
    out = (baseline(a.repo, a.name, ROOT / a.dest, a.untracked) if a.cmd == "baseline"
           else candidate(a.repo, a.name))
    print(json.dumps(out, indent=1))
