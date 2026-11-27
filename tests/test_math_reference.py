import unittest

from validation.math_reference import (
    kda_chunked_equals_single_pass, mla_prefill_vs_absorbed,
)


class MathReferenceTests(unittest.TestCase):
    def test_mla_prefill_mha_equals_absorbed_latent(self):
        for seed in range(5):
            self.assertLess(mla_prefill_vs_absorbed(seed=seed), 1e-10)

    def test_kda_fixed_size_state_suffices_across_chunks(self):
        for seed in range(5):
            self.assertLess(kda_chunked_equals_single_pass(seed=seed), 1e-10)

    def test_kda_negative_controls_detect_missing_state(self):
        self.assertGreater(kda_chunked_equals_single_pass(carry_conv=False), 1e-3)
        self.assertGreater(kda_chunked_equals_single_pass(carry_state=False), 1e-3)


if __name__ == "__main__":
    unittest.main()
