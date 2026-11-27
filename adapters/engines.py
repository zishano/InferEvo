"""Engine locations, exact source identity and import access.

`engines/simthesizer` and `engines/rooflang` are git worktrees (branches
`inferevo/*`) of the user's checkouts `../simthesizer` and `../rooflang`,
which are never modified. Every result records `source_identity()`.
Environment overrides: INFEREVO_SIMTHESIZER, INFEREVO_ROOFLANG.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ROOT / "engines"
REFERENCES = ROOT / "references"
USER_SIMTHESIZER = ROOT.parent / "simthesizer"
USER_ROOFLANG = ROOT.parent / "rooflang"
SIMTHESIZER = Path(os.environ.get("INFEREVO_SIMTHESIZER", ENGINES / "simthesizer"))
ROOFLANG = Path(os.environ.get("INFEREVO_ROOFLANG", ENGINES / "rooflang"))


@dataclass(frozen=True)
class RepoIdentity:
    name: str
    path: str
    commit: str
    branch: str
    dirty: bool
    # sha256 of `git diff HEAD` plus untracked file names; empty when clean.
    dirty_digest: str


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def repo_identity(name: str, repo: Path) -> RepoIdentity:
    commit = _git(repo, "rev-parse", "HEAD").strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    diff = _git(repo, "diff", "HEAD", "--binary")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    dirty = bool(diff or untracked)
    digest = ""
    if dirty:
        h = hashlib.sha256(diff.encode())
        h.update(b"\0untracked\0" + untracked.encode())
        digest = h.hexdigest()
    return RepoIdentity(name, str(repo), commit, branch, dirty, digest)


def source_identity() -> dict:
    """Identity of every repository an experiment result depends on."""
    out = {}
    for name, repo in (("simthesizer", SIMTHESIZER), ("rooflang", ROOFLANG)):
        out[name] = asdict(repo_identity(name, repo))
    try:
        out["inferevo"] = asdict(repo_identity("inferevo", ROOT))
    except subprocess.CalledProcessError:
        out["inferevo"] = {"name": "inferevo", "commit": "uncommitted"}
    return out


def enable_bridge_imports() -> None:
    """Import Simthesizer's vendored RoofLang and its bridge adapters.

    These are the modules the simulator executes, so analyses price exactly
    what the simulator runs.
    """
    scripts = str(SIMTHESIZER / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
