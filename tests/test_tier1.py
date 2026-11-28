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


class NnlsTests(unittest.TestCase):
    def test_matches_unconstrained_when_feasible_and_clips_otherwise(self):
        import numpy as np
        from adapters.tier1 import nnls
        rng = np.random.default_rng(0)
        a = rng.random((40, 4)) * np.array([1, 10, 1e4, 1e8])       # very different column scales
        x_true = np.array([2.0, 0.5, 3e-3, 1e-7])
        b = a @ x_true
        np.testing.assert_allclose(nnls(a, b), x_true, rtol=1e-6)
        x_neg = np.array([2.0, -0.5, 3e-3, 1e-7])                     # infeasible truth -> clipped
        got = nnls(a, a @ x_neg)
        self.assertTrue((got >= 0).all())
        self.assertLess(np.linalg.norm(a @ got - a @ x_neg),
                        np.linalg.norm(a @ np.clip(x_neg, 0, None) - a @ x_neg) + 1e-9)
