"""Component registry, dependency graph, impact analysis and regression
selection (D7; spec 8.1/8.3).

A component is one source file of a runtime tree:
  sim:<path>   Simthesizer Rust (crates/...), Python scripts, launcher .sh
  rl:<path>    RoofLang package (engines/rooflang)
  ie:<path>    InferEvo (adapters/, agent/, tools/, validation/, mechanisms/)
The graph is scanned, not hand-maintained:
  Rust    `crate::` / `super::` / `simthesizer_core::` paths -> module files,
          `mod x;` -> child, lib.rs re-exports resolved; `register_*!(T, "n")`
          gives the component its registry kind and name; the CLI links every
          registered kind (link-time inventory).
  Python  `import` / `from` (ast), resolved against each tree's import roots.
  Shell   references to scripts/*.py.
  Binary  any file naming target/{debug,release}/simthesizer or SIMTHESIZER_BIN
          depends on the CLI component.
`components.toml` adds what scanning cannot see (cross-language process
edges) and curated facts per component (role, state owned, invariants).

Tests are components too: Rust unit tests (in-file #[cfg(test)]), Rust
integration tests, Simthesizer / RoofLang / InferEvo Python tests. Impact
of a change = reverse-dependency closure of the changed files; selected
regression tests = tests inside that closure.

    ./ie.sh -m mechanisms.registry show sim:crates/simthesizer-core/src/kv.rs
    ./ie.sh -m mechanisms.registry impact --repo simthesizer --base <commit> [--head <rev>]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from adapters.engines import ROOFLANG, ROOT, SIMTHESIZER

OVERLAY = Path(__file__).with_name("components.toml")
CORE = "crates/simthesizer-core/src"
CLI = "sim:crates/simthesizer-cli/src/main.rs"
REGISTER = re.compile(r"register_(system|scheduler|router|model_runner|model|compute_simulator|network_simulator)!\(\s*(\w+)\s*,\s*\"([^\"]+)\"")
RUST_PATH = re.compile(r"\b(crate|super|simthesizer_core)((?:::[A-Za-z_][A-Za-z0-9_]*)+)")
RUST_GROUP = re.compile(r"\b(crate|simthesizer_core)::\{([^}]*)\}", re.S)
MOD_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.M)
BINARY = re.compile(r'target/(?:debug|release)/simthesizer|SIMTHESIZER_BIN|"(?:debug|release)"\s*/\s*"simthesizer"|\["cargo",')
SKIP = {"__pycache__", "target", ".venv", "node_modules", ".git", "output", "experiments", "candidates"}


@dataclass
class Component:
    id: str
    repo: str
    path: Path                      # absolute
    lang: str
    kind: str = "module"            # registry kind, or "test"
    names: list = field(default_factory=list)   # registered names
    deps: set = field(default_factory=set)
    test: dict | None = None        # how to run it, for test components
    facts: dict = field(default_factory=dict)


TREES = {  # prefix: (repo, root, python import roots relative to root, globs)
    "sim": ("simthesizer", SIMTHESIZER, ("scripts", "."), ("crates/**/*.rs", "scripts/**/*.py", "tests/*.py", "*.sh")),
    "rl": ("rooflang", ROOFLANG, (), ("**/*.py",)),
    "ie": ("inferevo", ROOT, (".",), ("adapters/*.py", "agent/*.py", "tools/*.py", "validation/*.py",
                                      "mechanisms/*.py", "tests/*.py")),
}


def _files(root: Path, globs):
    for g in globs:
        for p in sorted(root.glob(g)):
            if p.is_file() and not SKIP & set(p.relative_to(root).parts):
                yield p


def _rust_module(rel: str) -> tuple[str, ...] | None:
    """crates/simthesizer-core/src/a/b.rs -> ("a", "b"); mod.rs/lib.rs -> its dir."""
    if not rel.startswith(CORE + "/"):
        return None
    parts = rel[len(CORE) + 1:-3].split("/")
    if parts[-1] in ("mod", "lib"):
        parts = parts[:-1]
    return tuple(parts)


class Registry:
    def __init__(self, roots: dict | None = None):
        """roots: {"sim": path, "rl": path, "ie": path} overrides (candidate trees)."""
        self.trees = {k: (repo, Path((roots or {}).get(k, root)), r, g) for k, (repo, root, r, g) in TREES.items()}
        self.c: dict[str, Component] = {}
        self._idx = {}
        for prefix, (repo, root, _, globs) in self.trees.items():
            for p in _files(root, globs):
                rel = str(p.relative_to(root))
                lang = {".rs": "rust", ".py": "python", ".sh": "shell"}[p.suffix]
                self.c[f"{prefix}:{rel}"] = Component(f"{prefix}:{rel}", repo, p, lang)
        self.modules = {m: cid for cid, comp in self.c.items()
                        if (m := _rust_module(cid[4:]) if cid.startswith("sim:") else None) is not None}
        self.reexports = self._reexports()
        for comp in self.c.values():
            getattr(self, f"_scan_{comp.lang}")(comp)
        registered = [cid for cid, comp in self.c.items() if comp.names and comp.kind != "test"]
        if CLI in self.c:
            self.c[CLI].deps |= set(registered)
        self._overlay()

    # ---------------- scanning ----------------
    def _reexports(self):
        lib = self.trees["sim"][1] / CORE / "lib.rs"
        out = {}
        for a, b in re.findall(r"pub use (\w+)::(\w+);", lib.read_text()):
            if b[0].islower():
                out[b] = (a, b)
        return out

    def _resolve_rust(self, base: tuple, segs: list[str]) -> str | None:
        if segs and segs[0] in self.reexports and not base:
            segs = [*self.reexports[segs[0]], *segs[1:]]
        for n in range(len(segs), 0, -1):
            cid = self.modules.get((*base, *segs[:n]))
            if cid:
                return cid
        return self.modules.get(base) if base else None

    def _scan_rust(self, comp: Component):
        rel = comp.id[4:]
        text = comp.path.read_text(errors="replace")
        mod = _rust_module(rel)
        integration = rel.startswith("crates/simthesizer-core/tests/")
        for kind, _, name in REGISTER.findall(text):
            comp.kind, comp.names = ("test" if integration else kind), [*comp.names, name]
        refs = [(root, path.split("::")[1:]) for root, path in RUST_PATH.findall(text)]
        for root, group in RUST_GROUP.findall(text):
            refs += [(root, item.strip().split("::")) for item in group.split(",") if item.strip()]
        for root, segs in refs:
            if root == "super" and mod:
                cid = self._resolve_rust(mod[:-1] if not rel.endswith("mod.rs") else mod[:-1], segs)
            elif root in ("crate", "simthesizer_core"):
                cid = self._resolve_rust((), segs)
            else:
                cid = None
            if cid and cid != comp.id:
                comp.deps.add(cid)
        for script in re.findall(r"(\w+)\.py\b", text):
            if f"sim:scripts/{script}.py" in self.c:
                comp.deps.add(f"sim:scripts/{script}.py")
        if mod is not None:
            for child in MOD_DECL.findall(text):
                cid = self.modules.get((*mod, child))
                if cid:
                    comp.deps.add(cid)
        if mod is not None and "#[cfg(test)]" in text:
            comp.test = {"cmd": ["cargo", "test", "--locked", "-p", "simthesizer-core", "--lib", "::".join(mod) + "::"],
                         "cwd": "sim", "kind": "rust-unit"}
        if integration and rel.count("/") == 3:
            comp.kind = "test"
            comp.test = {"cmd": ["cargo", "test", "--locked", "-p", "simthesizer-core", "--test", comp.path.stem],
                         "cwd": "sim", "kind": "rust-integration"}
            common = self.c.get("sim:crates/simthesizer-core/tests/common/mod.rs")
            if "mod common" in text and common:
                comp.deps.add(common.id)
        if rel.startswith("crates/simthesizer-cli/tests/"):
            comp.kind = "test"
            comp.deps.add(CLI)
            comp.test = {"cmd": ["cargo", "test", "--locked", "-p", "simthesizer-cli", "--test", comp.path.stem],
                         "cwd": "sim", "kind": "rust-integration"}

    def _py_index(self, prefix):
        if prefix in self._idx:
            return self._idx[prefix]
        repo, root, roots, _ = self.trees[prefix]
        idx = {}
        for cid, comp in self.c.items():
            if not cid.startswith(prefix + ":") or comp.lang != "python":
                continue
            rel = comp.path.relative_to(root)
            for r in roots:
                try:
                    sub = rel.relative_to(r) if r != "." else rel
                except ValueError:
                    continue
                parts = list(sub.with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts:
                    idx.setdefault(".".join(parts), cid)
        if prefix == "rl":   # the package is imported as `rooflang.*`
            for cid, comp in self.c.items():
                if cid.startswith("rl:"):
                    parts = list(comp.path.relative_to(root).with_suffix("").parts)
                    if parts[-1] == "__init__":
                        parts = parts[:-1]
                    idx[".".join(["rooflang", *parts])] = cid
        self._idx[prefix] = idx
        return idx

    def _resolve_py(self, prefix, name):
        # The simulator (and InferEvo via enable_bridge_imports) runs the
        # RoofLang copy vendored in simthesizer/scripts; engines/rooflang is
        # the standalone tree with its own tests.
        for pre in {"sim": ("sim",), "rl": ("rl",), "ie": ("ie", "sim")}[prefix]:
            idx = self._py_index(pre)
            parts = name.split(".")
            for n in range(len(parts), 0, -1):
                cid = idx.get(".".join(parts[:n]))
                if cid:
                    return cid
        return None

    def _scan_python(self, comp: Component):
        prefix = comp.id.split(":", 1)[0]
        text = comp.path.read_text(errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level:
                base = comp.path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                mod = (node.module or "").replace(".", "/")
                for cand in (base / mod / "__init__.py", (base / mod).with_suffix(".py"),
                             *[(base / mod / a.name).with_suffix(".py") for a in node.names]):
                    cid = next((k for k, v in self.c.items() if v.path == cand), None)
                    if cid:
                        comp.deps.add(cid)
        for n in names:
            cid = self._resolve_py(prefix, n)
            if cid and cid != comp.id:
                comp.deps.add(cid)
        if BINARY.search(text) and CLI in self.c:
            comp.deps.add(CLI)
        for script in re.findall(r"scripts/(\w+)\.py", text):
            cid = f"sim:scripts/{script}.py"
            if cid in self.c and cid != comp.id:
                comp.deps.add(cid)
        for launcher in re.findall(r"\b(run\w*\.sh)\b", text):
            if f"sim:{launcher}" in self.c:
                comp.deps.add(f"sim:{launcher}")
        if comp.path.name.startswith("test_"):
            comp.kind = "test"
            if prefix == "sim" and re.search(r"^import test_\w+", text, re.M):
                # imports a sibling test module: run from tests/
                comp.test = {"cmd": [".venv/bin/python", "-m", "unittest", comp.path.stem],
                             "cwd": "sim-tests", "kind": "python"}
            elif prefix == "sim":
                comp.test = {"cmd": [".venv/bin/python", "-m", "unittest", f"tests.{comp.path.stem}"],
                             "cwd": "sim", "kind": "python"}
            elif prefix == "rl":
                rl = self.trees["rl"][1]
                comp.test = {"cmd": [str(SIMTHESIZER / ".venv/bin/python"), "-m", "pytest", "-q",
                                     str(comp.path.relative_to(rl.parent))], "cwd": str(rl.parent),
                             "kind": "python", "env": {"PYTHONPATH": str(rl.parent)}}
            else:
                comp.test = {"cmd": ["./ie.sh", "-m", "unittest", f"tests.{comp.path.stem}"],
                             "cwd": "ie", "kind": "python"}

    def _scan_shell(self, comp: Component):
        text = comp.path.read_text(errors="replace")
        for script in re.findall(r"scripts/(\w+)\.py", text):
            cid = f"sim:scripts/{script}.py"
            if cid in self.c:
                comp.deps.add(cid)
        if "cargo" in text or BINARY.search(text):
            comp.deps.add(CLI)

    def _overlay(self):
        if not OVERLAY.exists():
            return
        data = tomllib.loads(OVERLAY.read_text())
        for cid, facts in data.get("component", {}).items():
            if cid not in self.c:
                raise KeyError(f"components.toml names unknown component {cid}")
            comp = self.c[cid]
            for d in facts.get("depends_on", []):
                if d not in self.c:
                    raise KeyError(f"components.toml: {cid} depends on unknown {d}")
                comp.deps.add(d)
            comp.facts = {k: v for k, v in facts.items() if k != "depends_on"}

    # ---------------- queries ----------------
    def reverse(self):
        rev = defaultdict(set)
        for cid, comp in self.c.items():
            for d in comp.deps:
                rev[d].add(cid)
        return rev

    def closure(self, changed: set[str]) -> set[str]:
        rev, seen, todo = self.reverse(), set(changed), deque(changed)
        while todo:
            for up in rev[todo.popleft()] - seen:
                seen.add(up)
                todo.append(up)
        return seen

    def tests_for(self, changed: set[str]) -> list[Component]:
        return sorted((self.c[cid] for cid in self.closure(changed) if self.c[cid].test),
                      key=lambda c: (c.test["kind"], c.id))

    def from_files(self, repo: str, paths) -> tuple[set[str], list[str]]:
        prefix = {"simthesizer": "sim", "rooflang": "rl", "inferevo": "ie"}[repo]
        known, unknown = set(), []
        for p in paths:
            cid = f"{prefix}:{p}"
            (known.add(cid) if cid in self.c else unknown.append(p))
        return known, unknown

    def topological(self, ids) -> list[str]:
        """Extension order: dependencies before dependents (spec 8: planner)."""
        ids, out, mark = set(ids), [], {}
        def visit(n):
            if mark.get(n) == 2:
                return
            if mark.get(n) == 1:
                return                      # cycle: keep scan order
            mark[n] = 1
            for d in sorted(self.c[n].deps & ids):
                visit(d)
            mark[n] = 2
            out.append(n)
        for n in sorted(ids):
            visit(n)
        return out

    def describe(self, cid: str) -> dict:
        comp = self.c[cid]
        return {"id": cid, "repo": comp.repo, "lang": comp.lang, "kind": comp.kind, "registered": comp.names,
                "depends_on": sorted(comp.deps), "used_by": sorted(self.reverse()[cid]),
                "test": comp.test, **comp.facts}


def changed_files(repo_path: Path, base: str, head: str | None = None) -> list[str]:
    args = ["git", "-C", str(repo_path), "diff", "--name-only", base] + ([head] if head else [])
    return [l for l in subprocess.run(args, check=True, capture_output=True, text=True).stdout.splitlines() if l]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("ids", nargs="+")
    i = sub.add_parser("impact")
    i.add_argument("--repo", choices=("simthesizer", "rooflang", "inferevo"), required=True)
    i.add_argument("--path", type=Path, default=None, help="checkout to diff (default: the engine tree)")
    i.add_argument("--base", required=True)
    i.add_argument("--head", default=None)
    sub.add_parser("stats")
    a = p.parse_args()
    reg = Registry()
    if a.cmd == "show":
        print(json.dumps([reg.describe(c) for c in a.ids], indent=1))
    elif a.cmd == "stats":
        kinds = defaultdict(int)
        for comp in reg.c.values():
            kinds[(comp.repo, comp.kind)] += 1
        print(json.dumps({f"{r}/{k}": n for (r, k), n in sorted(kinds.items())}, indent=1))
        print("edges", sum(len(c.deps) for c in reg.c.values()))
    else:
        root = a.path or {"simthesizer": SIMTHESIZER, "rooflang": ROOFLANG, "inferevo": ROOT}[a.repo]
        known, unknown = reg.from_files(a.repo, changed_files(root, a.base, a.head))
        tests = reg.tests_for(known)
        print(json.dumps({"changed": sorted(known), "untracked_by_registry": unknown,
                          "impact": sorted(reg.closure(known)), "tests": [t.id for t in tests]}, indent=1))


if __name__ == "__main__":
    main()
