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

    def test_nearestValidPoint(self):
        # 1779.Find Nearest Point That Has the Same X or Y Coordinate
        points = [[1, 2], [3, 1], [2, 4], [2, 3], [4, 4]]
        self.assertEqual(self.sl.nearestValidPoint(3, 4, points), 2)
        self.assertEqual(self.sl.nearestValidPoint(3, 4, [[3, 4]]), 0)
        self.assertEqual(self.sl.nearestValidPoint(3, 4, [[2, 3]]), -1)

    def test_areAlmostEqual(self):
        # 1790.Check if One String Swap Can Make Strings Equal
        self.assertTrue(self.sl.areAlmostEqual("bank", "kanb"))
        self.assertTrue(self.sl.areAlmostEqual("kelb", "kelb"))
        self.assertFalse(self.sl.areAlmostEqual("attack", "defend"))
        self.assertFalse(self.sl.areAlmostEqual("yhy", "hyc"))
