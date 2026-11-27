import unittest

from adapters.topologies import Deployment
from agent.controller import dominates, neighbours, rule_policy


class ControllerTests(unittest.TestCase):
    def test_dominance_uses_objective_directions(self):
        a = {"tokens_per_s_gpu": 30, "ttft_p99_ms": 100, "tpot_p50_ms": 10}
        b = {"tokens_per_s_gpu": 25, "ttft_p99_ms": 120, "tpot_p50_ms": 10}
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))
        self.assertFalse(dominates(a, a))

    def test_prefill_bottleneck_proposes_budget_before_instance_shift(self):
        dep = Deployment(prefill_instances=1, decode_instances=3)
        entry = {"deployment": dep.__dict__, "bottlenecks": [{"class": "prefill_saturated"}]}
        moves = [d for d, _ in rule_policy(entry)]
        self.assertEqual(moves[0].prefill_budget, 512)
        self.assertEqual((moves[1].prefill_instances, moves[1].decode_instances), (2, 2))

    def test_neighbours_keep_gpu_count(self):
        dep = Deployment(prefill_instances=2, decode_instances=2, prefill_budget=512)
        self.assertTrue(all(n.gpus == dep.gpus for n, _ in neighbours(dep)))
        self.assertEqual(len(neighbours(dep)), 5)


if __name__ == "__main__":
    unittest.main()
