import unittest

from validation.round_bounds import isolated_time, round_bounds


def op(i, parents, demands, slots=(), compute_us=None):
    work = {"demands": demands, "slots": list(slots), "details": {"name": f"op{i}"}}
    return {"id": i, "parents": parents, "work": work}


class RoundBoundsTests(unittest.TestCase):
    def test_isolated_time_is_slowest_component_after_latency(self):
        work = {"demands": [{"resources": {"a": 2.0, "b": 1.0}, "latency": 0.5},
                            {"resources": {"c": 1.0}}]}
        self.assertEqual(isolated_time(work), 2.5)

    def test_bounds_cover_critical_path_resource_load_and_streams(self):
        ops = [op(0, [], [{"resources": {"hbm:g0": 1.0}}], ["s0"]),
               op(1, [], [{"resources": {"hbm:g0": 1.0}}], ["s1"]),
               op(2, [0, 1], [{"resources": {"compute:g0": 0.5}}], ["s0"])]
        b = round_bounds(ops, "h200")
        self.assertEqual(b.critical_path_s, 1.5)
        self.assertEqual(b.max_resource_load_s, 2.0)   # shared HBM serializes
        self.assertEqual(b.lower_s, 2.0)
        self.assertEqual(b.upper_s, 2.5)
        self.assertEqual(b.check(2.5), [])
        self.assertTrue(b.check(1.9))  # faster than physically possible
        self.assertTrue(b.check(2.6))  # slower than fully serial

    def test_pricing_identity_detects_wrong_compute_time(self):
        details = {"name": "gemm", "flops_by_dtype": {"bf16": 989.5e6}, "compute_us": 2.0}
        o = op(0, [], [{"resources": {"compute:g0": 2e-6}}])
        o["work"]["details"] = details
        self.assertTrue(round_bounds([o], "h200").pricing_errors)  # expected 1.0 us
        details["compute_us"] = 1.0
        o["work"]["demands"][0]["resources"]["compute:g0"] = 1e-6
        self.assertEqual(round_bounds([o], "h200").pricing_errors, [])


if __name__ == "__main__":
    unittest.main()
