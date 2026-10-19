"""Check the split and metric properties that protect evaluation validity."""

import unittest

import numpy as np

from experiment import normalized_rmse, split_indices


class ProtocolTests(unittest.TestCase):
    def test_split_is_disjoint_reproducible_and_60_20_20(self):
        first = split_indices(1000, 42)
        second = split_indices(1000, 42)
        self.assertEqual([len(x) for x in first], [600, 200, 200])
        self.assertEqual(len(set(np.concatenate(first))), 1000)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_nrmse_uses_given_training_scale(self):
        actual = np.array([2.0, 4.0])
        predicted = np.array([3.0, 5.0])
        self.assertAlmostEqual(normalized_rmse(actual, predicted, train_target_sd=2.0), 0.5)
        with self.assertRaises(ValueError):
            normalized_rmse(actual, predicted, train_target_sd=0.0)


if __name__ == "__main__":
    unittest.main()

