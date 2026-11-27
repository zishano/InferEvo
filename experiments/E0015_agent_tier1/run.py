"""E0015: long-context Tier-1 refit + agent-session validation (H200 TP4).

1. Decode surrogate on graph bounds with kv up to 32k.
2. Prefill surrogate on engine samples: E0012 samples + long single prompts
   (1.1k-16k tokens, TTFT = sum of 128-token chunk rounds).
3. Agent sessions (6 x 4 turns, closed loop, contexts to ~10k) through
   agent_dag over P/D at Tier-2 (shared Rooflang) and Tier-1 (step_model);
   compare session completion time (JCT) and per-call TTFT/TPOT.

    ./ie.sh experiments/E0015_agent_tier1/run.py
"""
import json, random, statistics, subprocess, tempfile, time
from dataclasses import asdict
from pathlib import Path
from adapters import tier1, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import ROOT, SIMTHESIZER, source_identity

here = Path(__file__).parent
E = ROOT / "experiments"
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
base_cfg = te.load(E / "E0008_tier1_feasibility/config.toml")
log = lambda *a: print(*a, flush=True)

# 1. decode
dec_path = here / "decode_model.json"
if dec_path.exists():
    dec = tier1.Tier1Model(**json.loads(dec_path.read_text()))
else:
    dec = tier1.fit("h200", 4, 16, "d", samples=120, seed=2, max_requests=32, max_kv=32768)
    dec_path.write_text(json.dumps(asdict(dec)))
log("decode fit", dec.train_median_rel_err, dec.train_p90_rel_err)

# 2. prefill: reuse E0012 samples + long prompts
cache = here / "long_prefill_samples.jsonl"
rng = random.Random(5)
lengths = [rng.randint(1100, 16384) for _ in range(12)]
done = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
subprocess.run(["cargo", "build", "--release", "--locked", "-q"], cwd=SIMTHESIZER, check=True)
with tempfile.TemporaryDirectory(prefix="e0015p-") as tmp:
    for k, length in enumerate(lengths):
        if k < len(done):
            continue
        rid = rng.randrange(1000)
        wl = workloads.write([{"request_id": rid, "arrival_time": 1.0, "initial": [0], "sub_requests": [
            {"kind": "llm", "input_tokens": length, "output_tokens": 2, "next": []}]}], Path(tmp) / f"w{k}.jsonl")
        subprocess.run([str(BIN), "--config", str(E / "E0008_tier1_feasibility/config.toml"), "--workload", str(wl),
                        "--output-jsonl", str(Path(tmp) / f"r{k}.jsonl")], check=True, capture_output=True)
        ttft = json.loads((Path(tmp) / f"r{k}.jsonl").read_text())["sub_requests"][0]["token_latency"][0]
        rounds = [[(rid, min(128, length - d), d)] for d in range(0, length, 128)]
        row = {"index": 1000 + k, "kind": "B", "rounds": rounds, "engine_s": ttft}
        with cache.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.append(row)
        log(f"long prefill {length} tokens: ttft {ttft*1e3:.1f} ms")
short = [json.loads(l) for l in (E / "E0012_tier1_prefill_engine/samples.jsonl").read_text().splitlines()]
samples = short + done
key = {"hardware": "h200", "tp": 4, "ep": 16, "groups": 4, "stage": "p", "efficiency": {},
       "cost_model_version": cost_model_version(), "target": "Tier-2 engine TTFT/rounds, prompts to 16k"}
train = [s for s in samples if s["index"] % 5]
test = [s for s in samples if not s["index"] % 5]
pre = tier1.fit_round_sums([s["rounds"] for s in train], [s["engine_s"] for s in train], key)
pre_err = [abs(sum(pre.step_time(b) for b in s["rounds"]) - s["engine_s"]) / s["engine_s"] for s in test]
log("prefill held-out median/max", statistics.median(pre_err), max(pre_err))


# 3. agent sessions Tier-2 vs Tier-1
def jct_and_calls(path, rows):
    arrival = {r["request_id"]: r["arrival_time"] for r in rows}
    jct, calls = {}, {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        lat = [s for s in r["sub_requests"] if s["kind"] == "llm"]
        calls[r["request_id"]] = [(s["token_latency"][0], statistics.fmean(s["token_latency"][1:])) for s in lat]
        jct[r["request_id"]] = None  # filled from DAG timing below
        # closed-loop chain: completion = arrival + sum of LLM latency + tool durations + intervals
        t = arrival[r["request_id"]]
        for s in r["sub_requests"]:
            t += (s.get("interval") or 0.0) + (sum(s["token_latency"]) if s["kind"] == "llm" else s["duration"])
        jct[r["request_id"]] = t - arrival[r["request_id"]]
    return jct, calls


rows = workloads.agent_sessions(6, turns=4, rate=0.5, seed=3, tool_output=(100, 1500))
out = {}
with tempfile.TemporaryDirectory(prefix="e0015a-") as tmp:
    wl = workloads.write(rows, Path(tmp) / "agents.jsonl")
    t2cfg = {"system": {"kind": "agent_dag", "inner": base_cfg["system"]}}
    te.write_config(t2cfg, Path(tmp) / "t2.toml")
    t0 = time.perf_counter()
    subprocess.run([str(BIN), "--config", str(Path(tmp) / "t2.toml"), "--workload", str(wl),
                    "--output-jsonl", str(Path(tmp) / "t2.jsonl")], check=True, capture_output=True)
    wall2 = time.perf_counter() - t0
    (here / "tier2_agent_result.jsonl").write_text((Path(tmp) / "t2.jsonl").read_text())
    t1cfg = te.tier1_config(base_cfg, {"p": pre, "d": dec})
    t1cfg = {"system": {"kind": "agent_dag", "inner": t1cfg["system"]}}
    te.write_config(t1cfg, Path(tmp) / "t1.toml")
    t0 = time.perf_counter()
    subprocess.run([str(BIN), "--config", str(Path(tmp) / "t1.toml"), "--workload", str(wl),
                    "--output-jsonl", str(Path(tmp) / "t1.jsonl")], check=True, capture_output=True)
    wall1 = time.perf_counter() - t0
    j2, c2 = jct_and_calls(Path(tmp) / "t2.jsonl", rows)
    j1, c1 = jct_and_calls(Path(tmp) / "t1.jsonl", rows)
jct_err = [(j1[s] - j2[s]) / j2[s] for s in j2]
ttft_err = [(a[0] - b[0]) / b[0] for s in c2 for a, b in zip(c1[s], c2[s])]
tpot_err = [(a[1] - b[1]) / b[1] for s in c2 for a, b in zip(c1[s], c2[s])]
summary = {"jct": {"median_abs": statistics.median(abs(e) for e in jct_err), "max_abs": max(abs(e) for e in jct_err)},
           "call_ttft": {"median_abs": statistics.median(abs(e) for e in ttft_err), "max_abs": max(abs(e) for e in ttft_err)},
           "call_tpot": {"median_abs": statistics.median(abs(e) for e in tpot_err), "max_abs": max(abs(e) for e in tpot_err)},
           "tier2_mean_jct_s": statistics.fmean(j2.values()), "tier1_mean_jct_s": statistics.fmean(j1.values()),
           "tier2_wall_s": wall2, "tier1_wall_s": wall1}
log(json.dumps(summary, indent=1))
(here / "result.json").write_text(json.dumps({
    "experiment": "E0015", "decode": asdict(dec), "prefill": asdict(pre),
    "prefill_heldout": {"median": statistics.median(pre_err), "max": max(pre_err)},
    "agent_summary": summary, "source_identity": source_identity()}, indent=1))
