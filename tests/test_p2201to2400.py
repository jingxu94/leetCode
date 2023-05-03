import unittest

from pysolutions import Pro2201To2400


class TestP2201To2400(unittest.TestCase):
    @property
    def sl(self):
        return Pro2201To2400()

    def test_findDifference(self):
        # 2215.Find the Difference of Two Arrays
        self.assertEqual(self.sl.findDifference([1, 2, 3], [1, 2, 4]), [[3], [4]])
        self.assertEqual(self.sl.findDifference([1, 2, 3], [1, 2, 3]), [[], []])

    def test_maxValueOfCoins(self):
        # 2218.Maximum Value of K Coins From Piles
        self.assertEqual(self.sl.maxValueOfCoins([[1, 100, 3], [7, 8, 9]], 2), 101)
        self.assertEqual(
            self.sl.maxValueOfCoins([[100], [100], [100], [100], [100], [100], [1, 1, 1, 1, 1, 1, 700]], 7), 706
        )

    def test_successfulPairs(self):
        # 2300.Successful Pairs of Spells and Potions
        self.assertEqual(self.sl.successfulPairs([5, 1, 3], [1, 2, 3, 4, 5], 7), [4, 0, 3])
        self.assertEqual(self.sl.successfulPairs([3, 1, 2], [8, 5, 8], 16), [2, 0, 2])

    def test_countPairs(self):
        # 2316.Count Unreachable Pairs of Nodes in an Undirected Graph
        self.assertEqual(self.sl.countPairs(3, [[0, 1], [0, 2], [1, 2]]), 0)
        self.assertEqual(self.sl.countPairs(7, [[0, 2], [0, 5], [2, 4], [1, 6], [5, 4]]), 14)

    def test_zeroFilledSubarray(self):
        # 2348.Number of Zero-Filled Subarrays
        self.assertEqual(self.sl.zeroFilledSubarray([1, 3, 0, 0, 2, 0, 0, 4]), 6)
        self.assertEqual(self.sl.zeroFilledSubarray([0, 0, 0, 2, 0, 0]), 9)
        self.assertEqual(self.sl.zeroFilledSubarray([2, 10, 2019]), 0)

    def test_longestCycle(self):
        # 2360.Longest Cycle in a Graph
        self.assertEqual(self.sl.longestCycle([3, 3, 4, 2, 3]), 3)
        self.assertEqual(self.sl.longestCycle([2, -1, 3, 1]), -1)

    def test_removeStars(self):
        # 2390.Removing Stars From a String
        self.assertEqual(self.sl.removeStars("leet**cod*e"), "lecoe")
        self.assertEqual(self.sl.removeStars("erase*****"), "")
