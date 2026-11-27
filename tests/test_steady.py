import json
import tempfile
import unittest
from pathlib import Path

from adapters import steady, workloads


def _result(rows, ttft, tpot):
    return "".join(json.dumps({"request_id": r["request_id"], "arrival_time": r["arrival_time"], "sub_requests": [
        dict(s, token_latency=[ttft] + [tpot] * (s["output_tokens"] - 1)) for s in r["sub_requests"]]}) + "\n"
        for r in rows)


class SteadyTests(unittest.TestCase):
    def test_window_metrics_and_capacity(self):
        rows = workloads.tiled(3, rate=20.0, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.jsonl"
            p.write_text(_result(rows, 0.5, 0.01))
            m = steady.metrics(p, 16, {"ttft_s": 2.0, "tpot_s": 0.05})
        self.assertTrue(m["stable"])
        self.assertEqual(m["slo_attainment"], 1.0)
        self.assertAlmostEqual(m["offered_rate"], 20.0, delta=4.0)
        fake = lambda r: {"stable": True, "slo_attainment": 1.0 if r <= 7.3 else 0.5}
        self.assertAlmostEqual(steady.capacity(fake, 1.0, 4.0, steps=12)["rate"], 7.3, delta=0.05)

    def test_decode_backlog_is_a_queue(self):
        # Calls wait longer and longer for a decode slot (the 2nd-token gap):
        # TTFT and mean TPOT stay within SLO, yet the system is not stable.
        rows = workloads.fixed(90, rate=10.0, isl=100, osl=100, seed=3)
        body = "".join(json.dumps({"request_id": r["request_id"], "arrival_time": r["arrival_time"], "sub_requests": [
            dict(s, token_latency=[0.2, 0.01 + 0.05 * i] + [0.01] * 98) for s in r["sub_requests"]]}) + "\n"
            for i, r in enumerate(rows))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.jsonl"
            p.write_text(body)
            m = steady.metrics(p, 16, {"ttft_s": 2.0, "tpot_s": 0.05})
        self.assertEqual(m["queue"], "decode")
        self.assertFalse(m["stable"])
        self.assertEqual(m["ttft_violation"], 0.0)

    def test_zero_gap_tokens_invalidate_a_run(self):
        rows = workloads.fixed(60, rate=10.0, isl=100, osl=50, seed=4)
        body = "".join(json.dumps({"request_id": r["request_id"], "arrival_time": r["arrival_time"], "sub_requests": [
            dict(s, token_latency=[0.2] + [0.01] * 9 + [0.0] * 40) for s in r["sub_requests"]]}) + "\n" for r in rows)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.jsonl"
            p.write_text(body)
            m = steady.metrics(p, 16, {"ttft_s": 2.0, "tpot_s": 0.05})
        self.assertEqual(m["queue"], "invalid")
        self.assertFalse(steady.meets(m))

    def test_long_input_ranges(self):
        rows = workloads.long_input(200, rate=1.0, seed=2)
        ins = [r["sub_requests"][0]["input_tokens"] for r in rows]
        self.assertTrue(4096 <= min(ins) and max(ins) <= 32768)


if __name__ == "__main__":
    unittest.main()
