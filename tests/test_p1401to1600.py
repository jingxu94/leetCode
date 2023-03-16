import unittest

from pysolutions import Pro1401To1600


class TestP1401To1600(unittest.TestCase):
    @property
    def sl(self):
        return Pro1401To1600()

    def test_runningSum(self):
        # 1480.Running Sum of 1d Array
        self.assertEqual(self.sl.runningSum([1, 2, 3, 4]), [1, 3, 6, 10])
        self.assertEqual(self.sl.runningSum([1, 1, 1, 1, 1]), [1, 2, 3, 4, 5])
        self.assertEqual(self.sl.runningSum([3, 1, 2, 10, 1]), [3, 4, 6, 16, 17])

    def test_average(self):
        # 1491.Average Salary Excluding the Minimum and Maximum Salary
        self.assertEqual(self.sl.average([4000, 3000, 1000, 2000]), 2500.00)

    def test_countOdds(self):
        # 1523.Count Odd Numbers in an Interval Range
        self.assertEqual(self.sl.countOdds(3, 7), 3)
        self.assertEqual(self.sl.countOdds(8, 10), 1)

    def test_findKthPositive(self):
        # 1539.Kth Missing Positive Number
        self.assertEqual(self.sl.findKthPositive([2, 3, 4, 7, 11], 5), 9)
        self.assertEqual(self.sl.findKthPositive([1, 2, 3, 4], 2), 6)

    def test_maxNonOverlapping(self):
        # 1546.Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
        self.assertEqual(self.sl.maxNonOverlapping([1, 1, 1, 1, 1], 2), 2)
        self.assertEqual(self.sl.maxNonOverlapping([-1, 3, 5, 1, 4, 2, -9], 6), 2)
