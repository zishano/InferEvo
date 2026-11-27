import unittest

from adapters.topologies import Deployment
from agent import search
from agent.controller import models_for

PB = search.Problem()
DEP = Deployment(hardware="h200", prefill_instances=4, decode_instances=4, prefill_budget=512)


class DiagnoseTests(unittest.TestCase):
    def test_granularities(self):
        backlog = {"queue": "decode", "ttft_violation": 0.0, "tpot_violation": 0.0}
        self.assertEqual(search.diagnose(backlog, "queue"), "decode")
        # without the queue signal the decode backlog is invisible (D22)
        self.assertEqual(search.diagnose(backlog, "slo"), "prefill")
        self.assertEqual(search.diagnose(backlog, "none"), "unknown")
        self.assertEqual(search.diagnose(None, "queue"), "none")
        slo = {"queue": None, "ttft_violation": 0.1, "tpot_violation": 0.4}
        self.assertEqual(search.diagnose(slo, "queue"), "decode")


class MoveTests(unittest.TestCase):
    def test_directed_moves_come_first_and_are_deduped(self):
        order = [d for d, _ in search.ordered(PB, DEP, "prefill")]
        self.assertEqual(order[0].prefill_budget, 2048)
        self.assertEqual(len({d.key for d in order}), len(order))
        self.assertNotIn(DEP.key, {d.key for d in order})
        # an undirected list holds the same candidates
        self.assertEqual({d.key for d in order}, {d.key for d, _ in search.ordered(PB, DEP, "unknown")})

    def test_speedup_scales_one_side(self):
        base, fast = models_for(DEP), search._models(DEP, {"d": 2.0})
        batch = [(i, 1, 9000) for i in range(16)]
        self.assertAlmostEqual(fast["d"].step_time(batch), base["d"].step_time(batch) / 2)
        self.assertEqual(fast["p"].coef, base["p"].coef)


if __name__ == "__main__":
    unittest.main()
