import unittest

from pysolutions import Pro1601To1800


class TestP1601To1800(unittest.TestCase):
    @property
    def sl(self):
        return Pro1601To1800()

    def test_maximumWealth(self):
        # 1672.Richest Customer Wealth
        accounts1, accounts2, accounts3 = (
            [[1, 2, 3], [3, 2, 1]],
            [[1, 5], [7, 3], [3, 5]],
            [[2, 8, 7], [7, 1, 3], [1, 9, 5]],
        )
        res1, res2, res3 = 6, 10, 17
        self.assertEqual(res1, self.sl.maximumWealth(accounts1))
        self.assertEqual(res2, self.sl.maximumWealth(accounts2))
        self.assertEqual(res3, self.sl.maximumWealth(accounts3))
