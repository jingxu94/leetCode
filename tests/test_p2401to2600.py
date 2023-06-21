import unittest

from pysolutions import Pro2401To2600


class TestP2401To2600(unittest.TestCase):
    @property
    def sl(self):
        return Pro2401To2600()

    def test_partitionString(self):
        # 2405.Optimal Partition of String
        self.assertEqual(self.sl.partitionString("abacaba"), 4)
        self.assertEqual(self.sl.partitionString("ssssss"), 6)

    def test_minimizeArrayValue(self):
        # 2439.Minimize Maximum of Array
        self.assertEqual(self.sl.minimizeArrayValue([3, 7, 1, 6]), 5)
        self.assertEqual(self.sl.minimizeArrayValue([10, 1]), 10)

    def test_minCost(self):
        # 2448.Minimum Cost to Make Array Equal
        self.assertEqual(self.sl.minCost([1, 3, 5, 2], [2, 3, 1, 14]), 8)
        self.assertEqual(self.sl.minCost([2, 2, 2, 2, 2], [4, 2, 8, 1, 3]), 0)

    def test_totalCost(self):
        # 2462.Total Cost to Hire K Workers
        self.assertEqual(self.sl.totalCost([17, 12, 10, 2, 7, 2, 11, 20, 8], 3, 4), 11)
        self.assertEqual(self.sl.totalCost([1, 2, 4, 1], 3, 3), 4)
        self.assertEqual(self.sl.totalCost([31, 25, 72, 79, 74, 65, 84, 91, 18, 59, 27, 9, 81, 33, 17, 58], 11, 2), 423)

    def test_countGoodStrings(self):
        # 2466.Count Ways To Build Good Strings
        self.assertEqual(self.sl.countGoodStrings(1, 1, 1, 1), 2)
        self.assertEqual(self.sl.countGoodStrings(2, 2, 1, 1), 4)
        self.assertEqual(self.sl.countGoodStrings(3, 3, 1, 1), 8)
        self.assertEqual(self.sl.countGoodStrings(2, 3, 1, 2), 5)

    def test_minScore(self):
        # 2492.Minimum Score of a Path Between Two Cities
        self.assertEqual(self.sl.minScore(4, [[1, 2, 9], [2, 3, 6], [2, 4, 5], [1, 4, 7]]), 5)
        self.assertEqual(self.sl.minScore(4, [[1, 2, 2], [1, 3, 4], [3, 4, 7]]), 2)

    def test_maxScore(self):
        # 2542.Maximum Subsequence Score
        self.assertEqual(self.sl.maxScore([1, 3, 3, 2], [2, 1, 3, 4], 3), 12)
        self.assertEqual(self.sl.maxScore([4, 2, 3, 1, 1], [7, 5, 10, 9, 6], 1), 30)

    def test_splitNum(self):
        # 2578.Split With Minimum Sum
        self.assertEqual(self.sl.splitNum(4325), 59)
        self.assertEqual(self.sl.splitNum(687), 75)
