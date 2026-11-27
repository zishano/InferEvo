import json
import unittest
from pathlib import Path

from adapters import tier1


class Tier1Tests(unittest.TestCase):
    def test_features_follow_group_assignment(self):
        f = tier1.features([(0, 1, 10), (4, 1, 20), (1, 1, 5)], groups=4, stage="d")
        named = dict(zip(tier1.FEATURE_NAMES, f))
        self.assertEqual(named["max_group_requests"], 2)   # ids 0 and 4 share group 0
        self.assertEqual(named["max_group_kv"], 30)
        self.assertEqual(named["tokens"], 3)

    def test_recorded_decode_model_meets_error_budget(self):
        here = Path(__file__).parents[1] / "experiments" / "E0009_tier1_decode"
        r = json.loads(next(here.glob("result-*.json")).read_text())
        self.assertLess(r["engine_rounds"]["median_abs_rel_err"], 0.02)
        self.assertLess(r["engine_rounds"]["p90_abs_rel_err"], 0.06)
        model = tier1.Tier1Model(**r["model"])
        self.assertGreater(model.step_time([(0, 1, 100)]), 0.005)


if __name__ == "__main__":
    unittest.main()
