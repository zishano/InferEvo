"""E0012: Tier-1 prefill surrogate trained on Tier-2 *engine* rounds.

Prefill rounds exceed their graph lower bound (~1.7x at 128 tokens: MoE
dispatch/combine serialized on the group leader's NIC, E0011), so targets are
engine timings, observed without reports (metrics-only runs, seconds each):
  A) k requests (total <= 128 tokens) arriving together on an idle P
     instance form one round; each TTFT = that round.
  B) one request of L > 128 tokens runs back-to-back 128-token chunks;
     TTFT = sum of its chunk rounds (fitted on summed features).
H200 TP4/EP16 P1+D1, budget 128, output 2 tokens. Samples split 80/20.

    ./ie.sh experiments/E0012_tier1_prefill_engine/run.py
"""
import json, random, statistics, subprocess, tempfile
from dataclasses import asdict
from pathlib import Path
from adapters import tier1
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity

BUDGET, SEED, N_A, N_B = 128, 0, 40, 24
CACHE = Path(__file__).with_name("samples.jsonl")
rng = random.Random(SEED)


def ttft(rows, out):
    path = out.with_suffix(".jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run(["bash", "run_kimi_k3_shared.sh", "--workload", str(path), "--output-dir", str(out)],
                   cwd=SIMTHESIZER, check=True, capture_output=True)
    return {json.loads(l)["request_id"]: json.loads(l)["sub_requests"][0]["token_latency"][0]
            for l in (out / "result.jsonl").read_text().splitlines()}


def request(i, tokens):
    return {"request_id": i, "arrival_time": 1.0, "initial": [0],
            "sub_requests": [{"kind": "llm", "input_tokens": tokens, "output_tokens": 2, "next": []}]}


plans = []
for _ in range(N_A):
    k = rng.randint(1, 6)
    ids = rng.sample(range(1000), k)
    cuts = sorted(rng.sample(range(1, BUDGET), k - 1)) if k > 1 else []
    total = rng.randint(k, BUDGET)
    sizes = [max(1, round((b - a) * total / BUDGET)) for a, b in zip([0, *cuts], [*cuts, BUDGET])]
    plans.append(("A", ids, sizes))
for _ in range(N_B):
    plans.append(("B", [rng.randrange(1000)], [rng.randint(BUDGET + 1, 1100)]))

cached = [json.loads(l) for l in CACHE.read_text().splitlines()] if CACHE.exists() else []
samples = list(cached)
with tempfile.TemporaryDirectory(prefix="e0012-") as tmp:
    for idx, (kind, ids, sizes) in enumerate(plans):
        if idx < len(cached):
            continue
        t = ttft([request(i, q) for i, q in zip(ids, sizes)], Path(tmp) / f"o{idx}")
        if kind == "A":
            rounds = [[(i, q, 0) for i, q in zip(ids, sizes)]]
            target = statistics.fmean(t.values())
        else:
            (i,), (length,) = ids, sizes
            rounds, done = [], 0
            while done < length:
                q = min(BUDGET, length - done)
                rounds.append([(i, q, done)])
                done += q
            target = t[i]
        row = {"index": idx, "kind": kind, "rounds": rounds, "engine_s": target}
        with CACHE.open("a") as f:
            f.write(json.dumps(row) + "\n")
        samples.append(row)
        print(f"{idx} {kind} rounds={len(rounds)} ttft={target*1e3:.2f} ms", flush=True)
key = {"hardware": "h200", "tp": 4, "ep": 16, "groups": 4, "stage": "p", "efficiency": {},
       "cost_model_version": cost_model_version(), "target": "Tier-2 engine TTFT/rounds (E0012)"}
train = [s for s in samples if s["index"] % 5]
test = [s for s in samples if not s["index"] % 5]
model = tier1.fit_round_sums([s["rounds"] for s in train], [s["engine_s"] for s in train], key)
pred = lambda s: sum(model.step_time(b) for b in s["rounds"])
err = [abs(pred(s) - s["engine_s"]) / s["engine_s"] for s in test]
bound = tier1.Tier1Model(**json.load(open(sorted(Path(__file__).parents[1].glob(
    "E0010_tier1_prefill/result-*.json"))[-1]))["cases"][0]["model"])
bound_err = [(sum(bound.step_time(b) for b in s["rounds"]) - s["engine_s"]) / s["engine_s"] for s in samples]
out = {"model": asdict(model), "n_train": len(train), "n_test": len(test),
       "test_median_rel_err": statistics.median(err), "test_max_rel_err": max(err),
       "graph_bound_model_mean_signed_err": statistics.fmean(bound_err),
       "source_identity": source_identity()}
print(json.dumps({k: v for k, v in out.items() if k != "source_identity"}, indent=1))
(Path(__file__).with_name(f"result-{key['cost_model_version'][:12]}.json")).write_text(json.dumps(out, indent=1))
