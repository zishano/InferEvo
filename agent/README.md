# agent/

Research controller (ROADMAP Phase 3–4, DECISIONS D12): deterministic state
machine + append-only ledger driving agent work units (Codex/Claude CLI via
the engines' existing Docker launchers and skills). Not implemented yet.
Planned reuse: `engines/simthesizer/agents/qwen_moe_optimization` candidate
states, `agents/find_pareto_frontier` freezing/replay, RoofLang
`agents/find_new_arch` review protocol.

`llm_policy.py`: LLM-agent policy (same `policy(entry)` interface). The model
chooses only among numbered validated moves, sees the ledger, answers JSON;
failures fall back to the rules. `complete(prompt)` is injected — wiring a
real model service is pending the user's go-ahead (data leaves the machine).

Terminal interfaces (repo root):
- `./optimize.sh --gpus N --isl .. --osl .. [--ttft-slo --tpot-slo --evals --duration]`:
  agent search (tools/optimize.py) over P:D, TP, prefill budget and per-side
  layout {PP1xEP16, PP2xEP8, PP4xEP4} and decode batch {32, 64, 128} where
  Tier-1 surrogates pass D23; score = steady SLO capacity (D22 probes); ledger in
  experiments/_optimize/; prints best, Pareto (tok/s/GPU vs tok/s/user) and a
  reproduce command.
- `./simulate.sh --p --d [--tp --ep --etp --pp | --p-{tp,dp,ep,etp,pp,budget,max-batch}
  --d-{tp,dp,ep,etp,pp,budget,max-batch}] --isl --osl --rate [--capacity] [--tier 2]`:
  one simulation, per-side parallel parameters (tools/simulate.py). Both
  interfaces reject illegal layouts via `adapters.topologies.check`
  (launcher/engine rules, weights + KV memory); Tier-1 also requires a fitted
  surrogate (`models_for`), else `--tier 2`.
  Both report total tok/s/GPU (output tokens / all GPUs) and per side:
  prefill input tok/s per prefill GPU, decode output tok/s per decode GPU.
  `--p-moe-chunks` (M-MOECHUNK, Tier-1 for PP4xEP4 / PP2xEP8 at 8) is also
  searched; optimize.sh `--feedback` sets the agent's feedback granularity
  (D24, default resource).
- Workloads: `--workload agent [--turns 8]` (closed-loop coding sessions,
  capacity in sessions/s, session JCT); `simulate.sh --kv-retention`
  (M-KVRETAIN, Tier-1) keeps session prefixes on prefill instances; the agent
  search proposes it on agent workloads.
- Mechanisms (Phase 4, D25): `./ie.sh -m mechanisms.manage plan|new|check|
  evaluate|replay|promote|reject|status <ID>` over `mechanisms/specs/<ID>.toml`;
  `./ie.sh -m mechanisms.registry show|impact|stats` for components, impact
  and regression selection.
