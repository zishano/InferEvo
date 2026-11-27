"""E0014: Tier-1 model zoo — fit + end-to-end validation per layout.

For (hardware, attention TP) with EP16, P1+D1, peak profile:
  decode surrogate on graph bounds (E0009 method), prefill surrogate on
  engine TTFT samples (E0012 method), Tier-2 frozen-workload run, and the
  Tier-1 vs Tier-2 per-request differential (E0011 method).

    ./ie.sh experiments/E0014_tier1_zoo/run.py b300 4
"""
import json, random, statistics, subprocess, sys, tempfile, time
from dataclasses import asdict
from pathlib import Path
from adapters import tier1, tier1_engine as te
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity

HW, TP = sys.argv[1], int(sys.argv[2])
EP, BUDGET = 16, 128
here = Path(__file__).parent / f"{HW}-tp{TP}"
here.mkdir(exist_ok=True)
SHARED = ["--gpu-model", HW, "--prefill-gpu-model", HW, "--decode-gpu-model", HW,
          "--tp", str(TP), "--prefill-tp", str(TP), "--decode-tp", str(TP)]


def shared_run(script, workload, out):
    args = ["bash", script, "--output-dir", str(out), *SHARED]
    if workload:
        args += ["--workload", str(workload)]
    subprocess.run(args, cwd=SIMTHESIZER, check=True, capture_output=True)
    return out


def per_request(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        lat = r["sub_requests"][0]["token_latency"]
        out[r["request_id"]] = {"ttft": lat[0], "tpot": statistics.fmean(lat[1:]) if len(lat) > 1 else None,
                                "e2e": sum(lat)}
    return out


# 1. decode surrogate on graph bounds
dec = tier1.fit(HW, TP, EP, "d", samples=120, seed=0)
# 2. prefill surrogate on engine TTFT samples (cached)
cache = here / "prefill_samples.jsonl"
rng = random.Random(0)
plans = []
for _ in range(40):
    k = rng.randint(1, 6)
    ids = rng.sample(range(1000), k)
    cuts = sorted(rng.sample(range(1, BUDGET), k - 1)) if k > 1 else []
    total = rng.randint(k, BUDGET)
    plans.append(("A", ids, [max(1, round((b - a) * total / BUDGET)) for a, b in zip([0, *cuts], [*cuts, BUDGET])]))
for _ in range(24):
    plans.append(("B", [rng.randrange(1000)], [rng.randint(BUDGET + 1, 1100)]))
cached = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
samples = list(cached)
with tempfile.TemporaryDirectory(prefix="e0014-") as tmp:
    for idx, (kind, ids, sizes) in enumerate(plans):
        if idx < len(cached):
            continue
        wl = Path(tmp) / f"w{idx}.jsonl"
        wl.write_text("".join(json.dumps({"request_id": i, "arrival_time": 1.0, "initial": [0],
                                          "sub_requests": [{"kind": "llm", "input_tokens": q, "output_tokens": 2, "next": []}]}) + "\n"
                              for i, q in zip(ids, sizes)))
        t = {k: v["ttft"] for k, v in per_request(shared_run("run_kimi_k3_shared.sh", wl, Path(tmp) / f"o{idx}") / "result.jsonl").items()}
        if kind == "A":
            rounds, target = [[(i, q, 0) for i, q in zip(ids, sizes)]], statistics.fmean(t.values())
        else:
            (i,), (length,) = ids, sizes
            rounds = [[(i, min(BUDGET, length - d), d)] for d in range(0, length, BUDGET)]
            target = t[i]
        row = {"index": idx, "kind": kind, "rounds": rounds, "engine_s": target}
        with cache.open("a") as f:
            f.write(json.dumps(row) + "\n")
        samples.append(row)
        print(f"prefill sample {idx}: {target*1e3:.2f} ms", flush=True)
key = {"hardware": HW, "tp": TP, "ep": EP, "groups": EP // TP, "stage": "p", "efficiency": {},
       "cost_model_version": cost_model_version(), "target": "Tier-2 engine TTFT/rounds"}
train = [s for s in samples if s["index"] % 5]
test = [s for s in samples if not s["index"] % 5]
pre = tier1.fit_round_sums([s["rounds"] for s in train], [s["engine_s"] for s in train], key)
pre_err = [abs(sum(pre.step_time(b) for b in s["rounds"]) - s["engine_s"]) / s["engine_s"] for s in test]
# 3./4. frozen workload: Tier-2 vs Tier-1
with tempfile.TemporaryDirectory(prefix="e0014f-") as tmp:
    t0 = time.perf_counter()
    o2 = shared_run("run_kimi_k3_shared_workload.sh", None, Path(tmp) / "t2")
    wall2 = time.perf_counter() - t0
    t2 = per_request(o2 / "result.jsonl")
    (here / "tier2_frozen_result.jsonl").write_text((o2 / "result.jsonl").read_text())
    (here / "tier2_config.toml").write_text((o2 / "config.toml").read_text())
    cfg = te.tier1_config(te.load(o2 / "config.toml"), {"p": pre, "d": dec})
    te.write_config(cfg, Path(tmp) / "t1.toml")
    t0 = time.perf_counter()
    te.run(Path(tmp) / "t1.toml", SIMTHESIZER / "examples/workloads/realistic-10-seed42.jsonl", Path(tmp) / "t1.jsonl")
    wall1 = time.perf_counter() - t0
    t1 = per_request(Path(tmp) / "t1.jsonl")
diff = {}
for k in ("ttft", "tpot", "e2e"):
    e = [(t1[r][k] - t2[r][k]) / t2[r][k] for r in t2 if t2[r][k] is not None]
    diff[k] = {"median_abs": statistics.median(abs(x) for x in e), "max_abs": max(abs(x) for x in e),
               "mean_signed": statistics.fmean(e)}
out = {"hardware": HW, "tp": TP, "decode": asdict(dec), "prefill": asdict(pre),
       "prefill_test_median_rel_err": statistics.median(pre_err), "prefill_test_max_rel_err": max(pre_err),
       "e2e_differential": diff, "tier2_wall_s": wall2, "tier1_wall_s": wall1,
       "source_identity": source_identity()}
print(json.dumps({k: out[k] for k in ("prefill_test_median_rel_err", "prefill_test_max_rel_err", "e2e_differential", "tier2_wall_s", "tier1_wall_s")}, indent=1))
(here / "result.json").write_text(json.dumps(out, indent=1))
