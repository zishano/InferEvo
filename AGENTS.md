# InferEvo agent notes

Layout (responsibility per directory):

| Dir | Owns |
|---|---|
| `engines/` | runtime engines: git worktrees `simthesizer`, `rooflang` (branches `inferevo/*` of the user's `../simthesizer`, `../rooflang`) |
| `references/` | read-only upstream clones (symlinks into `../github/`) and paper notes; never on the runtime path |
| `adapters/` | engine access: locations, source identity, bridge imports, cost-model identity and queries |
| `mechanisms/` | candidate isolation (worktrees), version records, mechanism specs |
| `validation/` | independent checks: sourced hardware facts, model conformance, capacity |
| `agent/` | research controller (planned) |
| `experiments/` | `EXXXX/run.py` + version-keyed results |
| `docs/` | architecture, decisions, progress, source map, change records (`docs/changes/`) |

Rules:
- Goals/constraints: `RESEARCH_SPEC.md`. Resume from `docs/PROGRESS.md`;
  amend `docs/DECISIONS.md` with evidence rather than re-deciding.
- Never modify the user trees `../simthesizer`, `../rooflang` or their
  running processes. Engine changes are commits on `inferevo/*` branches.
- Cost-model or model-contract edits follow D6 (record in `docs/changes/`,
  re-evaluate baselines). Mechanism candidates live in `mechanisms/candidates/`.
- Every result records `source_identity()` and `cost_model_version()`;
  label claims with their evidence level (D11); never fabricate sources.
- Python: `./ie.sh` (engine venv, root on PYTHONPATH). Tests: `./run_tests.sh`.

File lifecycle (temporary files are deleted after use; files needed later are kept):
- Delete large report-mode outputs (`--reports`/`--full-graphs` traces,
  operator CSVs, graph HTML) as soon as their analysis is recorded.
- Temporary directories must be cleaned even on interruption (use
  `tempfile.TemporaryDirectory` / `trap ... EXIT`); after a crash or a
  stopped session, find and remove your own leftovers before continuing.
- Experiments keep only compact results (metrics, summaries, fitted models,
  version-keyed `result-*.json`). Keep Tier-2 `result.jsonl` files: they
  are reused for later refits and comparisons.
- Delete only temporary files you created yourself (scratchpad, your own
  temp dirs and run outputs). Never delete any other file: user data,
  user trees, references, other processes' outputs or anything you did
  not create. When unsure whether something is yours or still needed, keep
  it and ask.
