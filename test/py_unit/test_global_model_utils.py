import unittest

import numpy as np

from examples.global_model_utils import split_values, split_values_and_datetimes


class TestGlobalModelSplits(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.arange(20, dtype=np.float32).reshape(10, 2)
        dates = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
        self.datetimes = np.repeat(dates[:, None], 2, axis=1)

    def test_train_and_validation_can_use_all_values(self) -> None:
        train, validation, test = split_values(self.values, 0.9, 0.1)

        self.assertEqual(train.shape, (9, 2))
        self.assertEqual(validation.shape, (1, 2))
        self.assertEqual(test.shape, (0, 2))

    def test_paired_split_allows_an_empty_test_remainder(self) -> None:
        train, validation, test = split_values_and_datetimes(
            self.values, self.datetimes, 0.9, 0.1
        )

        self.assertEqual(train[0].shape, (9, 2))
        self.assertEqual(validation[0].shape, (1, 2))
        self.assertEqual(test[0].shape, (0, 2))
        np.testing.assert_array_equal(train[1], self.datetimes[:9])
        np.testing.assert_array_equal(validation[1], self.datetimes[9:])

    def test_test_remainder_is_still_supported(self) -> None:
        train, validation, test = split_values(self.values, 0.7, 0.1)

        self.assertEqual(train.shape, (7, 2))
        self.assertEqual(validation.shape, (1, 2))
        self.assertEqual(test.shape, (2, 2))

    def test_ratios_cannot_exceed_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            split_values(self.values, 0.9, 0.2)


if __name__ == "__main__":
    unittest.main()
