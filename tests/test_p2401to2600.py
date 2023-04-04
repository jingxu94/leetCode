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

    def test_minScore(self):
        # 2492.Minimum Score of a Path Between Two Cities
        self.assertEqual(self.sl.minScore(4, [[1, 2, 9], [2, 3, 6], [2, 4, 5], [1, 4, 7]]), 5)
        self.assertEqual(self.sl.minScore(4, [[1, 2, 2], [1, 3, 4], [3, 4, 7]]), 2)

    def test_splitNum(self):
        # 2578.Split With Minimum Sum
        self.assertEqual(self.sl.splitNum(4325), 59)
        self.assertEqual(self.sl.splitNum(687), 75)
