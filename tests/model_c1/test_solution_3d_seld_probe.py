import unittest

import numpy as np
import torch

from tools.clsdoa_v1.model_c1.probe_solution_3d_seld_input import (
    _SimpleRearrange,
    _simple_rearrange,
    array_stats,
)


class Solution3DSELDProbeTests(unittest.TestCase):
    def test_array_stats_records_shape_and_finite_counts(self):
        stats = array_stats(np.asarray([[1.0, np.nan], [np.inf, -2.0]], dtype=np.float32))
        self.assertEqual(stats["shape"], [2, 2])
        self.assertEqual(stats["dtype"], "float32")
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 1)
        self.assertFalse(stats["finite"])

    def test_local_einops_rearrange_patterns_used_by_solution_conformer(self):
        value = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
        split = _simple_rearrange(value, "b n (h d) -> b h n d", h=2)
        self.assertEqual(tuple(split.shape), (2, 2, 3, 4))
        merged = _simple_rearrange(split, "b h n d -> b n (h d)")
        self.assertTrue(torch.equal(merged, value))

        seq = torch.arange(5)
        self.assertEqual(tuple(_simple_rearrange(seq, "i -> i ()").shape), (5, 1))
        self.assertEqual(tuple(_simple_rearrange(seq, "j -> () j").shape), (1, 5))

    def test_local_rearrange_module_patterns_used_by_solution_conformer(self):
        value = torch.randn(2, 3, 4)
        to_channels = _SimpleRearrange("b n c -> b c n")
        to_time = _SimpleRearrange("b c n -> b n c")
        self.assertEqual(tuple(to_channels(value).shape), (2, 4, 3))
        self.assertTrue(torch.equal(to_time(to_channels(value)), value))


if __name__ == "__main__":
    unittest.main()
