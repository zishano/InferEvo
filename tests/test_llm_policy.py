import unittest

from adapters.topologies import Deployment
from agent.llm_policy import llm_policy, parse


class _Ctl:
    seen: dict = {}


ENTRY = {"key": "x", "deployment": Deployment().__dict__, "rationale": "baseline",
         "bottlenecks": [{"class": "prefill_saturated", "evidence": "busy 0.97"}],
         "objectives": {"tokens_per_s_gpu": 1, "ttft_p99_ms": 1, "tpot_p50_ms": 1}}


class LlmPolicyTests(unittest.TestCase):
    def test_choices_map_to_listed_options_in_order(self):
        prompts = []
        def complete(p):
            prompts.append(p)
            return 'ok [{"option": 1, "rationale": "b"}, {"option": 0, "rationale": "a"}]'
        moves = llm_policy(complete, _Ctl())(ENTRY)
        self.assertIn("Options:", prompts[0])
        self.assertEqual([m[1] for m in moves], ["llm: b", "llm: a"])

    def test_bad_answers_fall_back_to_rules(self):
        moves = llm_policy(lambda p: '[{"option": 99}]', _Ctl())(ENTRY)
        self.assertTrue(moves and all("llm fallback" in why for _, why in moves))
        with self.assertRaises(ValueError):
            parse("no json", 3)


if __name__ == "__main__":
    unittest.main()
