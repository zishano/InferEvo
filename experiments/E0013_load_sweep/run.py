"""E0013: load sweep at Tier-1 (H200 TP4/EP16 P1+D1, peak profile).

Realistic 100-request trace, arrivals compressed by rate multipliers.
Surrogates: E0009 decode (graph bound), E0012 prefill (engine rounds).
Throughput = output tokens / (last completion - first arrival) / 32 GPUs.
Evidence level: Tier-1 screening (surrogate error, see E0011); frontier
points are re-run at Tier-2 by confirm.py.

    ./ie.sh experiments/E0013_load_sweep/run.py
"""
import glob, json, statistics, tempfile, time
from pathlib import Path
from adapters import tier1, tier1_engine as te, workloads
from adapters.engines import ROOT, source_identity

E = ROOT / "experiments"
MULTIPLIERS = [0.5, 1, 2, 4, 8, 16]
GPUS = 32


def latest(pattern):
    return json.load(open(sorted(glob.glob(str(E / pattern)))[-1]))


def metrics(result_path, rows):
    arrivals = {r["request_id"]: r["arrival_time"] for r in rows}
    ttft, tpot, finish, tokens, users = [], [], [], 0, []
    for line in Path(result_path).read_text().splitlines():
        r = json.loads(line)
        (s,) = r["sub_requests"]
        lat = s["token_latency"]
        ttft.append(lat[0])
        if len(lat) > 1:
            tpot.append(statistics.fmean(lat[1:]))
            users.append((len(lat) - 1) / sum(lat[1:]))
        finish.append(arrivals[r["request_id"]] + sum(lat))
        tokens += len(lat)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
    span = max(finish) - min(arrivals.values())
    return {"ttft_ms": {"p50": q(ttft, .5) * 1e3, "p99": q(ttft, .99) * 1e3},
            "tpot_ms": {"p50": q(tpot, .5) * 1e3, "p99": q(tpot, .99) * 1e3},
            "tokens_per_s_gpu": tokens / span / GPUS,
            "tokens_per_s_user": statistics.fmean(users), "span_s": span}


def models():
    dec = tier1.Tier1Model(**latest("E0009_tier1_decode/result-*.json")["model"])
    pre = tier1.Tier1Model(**latest("E0012_tier1_prefill_engine/result-*.json")["model"])
    return {"p": pre, "d": dec}


if __name__ == "__main__":
    m = models()
    cfg = te.tier1_config(te.load(E / "E0008_tier1_feasibility/config.toml"), m)
    rows_out = []
    with tempfile.TemporaryDirectory(prefix="e0013-") as tmp:
        te.write_config(cfg, Path(tmp) / "config.toml")
        for mult in MULTIPLIERS:
            rows = workloads.scaled(mult)
            wl = workloads.write(rows, Path(tmp) / f"w{mult}.jsonl")
            t0 = time.perf_counter()
            te.run(Path(tmp) / "config.toml", wl, Path(tmp) / f"r{mult}.jsonl")
            res = metrics(Path(tmp) / f"r{mult}.jsonl", rows)
            res.update(multiplier=mult, offered_req_s=workloads.offered_rate(rows),
                       wall_s=time.perf_counter() - t0)
            rows_out.append(res)
            print(f"x{mult:<4} {res['offered_req_s']:6.1f} req/s  {res['tokens_per_s_gpu']:7.2f} tok/s/GPU "
                  f"{res['tokens_per_s_user']:6.1f} tok/s/user  TTFT p50/p99 {res['ttft_ms']['p50']:7.1f}/"
                  f"{res['ttft_ms']['p99']:7.1f} ms  TPOT p50/p99 {res['tpot_ms']['p50']:5.2f}/"
                  f"{res['tpot_ms']['p99']:5.2f} ms  wall {res['wall_s']:.1f}s", flush=True)
    (Path(__file__).with_name("result.json")).write_text(json.dumps({
        "experiment": "E0013", "tier": 1, "layout": "H200 TP4/EP16 P1+D1", "rows": rows_out,
        "models": {k: v.key for k, v in m.items()}, "source_identity": source_identity()}, indent=1))
