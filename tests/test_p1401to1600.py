import unittest

from pysolutions import Pro1401To1600


class TestP1401To1600(unittest.TestCase):
    @property
    def sl(self):
        return Pro1401To1600()

    def test_kidsWithCandies(self):
        # 1431.Kids With the Greatest Number of Candies
        self.assertEqual(self.sl.kidsWithCandies([2, 3, 5, 1, 3], 3), [True, True, True, False, True])
        self.assertEqual(self.sl.kidsWithCandies([4, 2, 1, 1, 2], 1), [True, False, False, False, False])
        self.assertEqual(self.sl.kidsWithCandies([12, 1, 12], 10), [True, False, True])

    def test_ways(self):
        # 1444.Number of Ways of Cutting a Pizza
        self.assertEqual(self.sl.ways(["A..", "AAA", "..."], 3), 3)
        self.assertEqual(self.sl.ways(["A..", "AA.", "..."], 3), 1)
        self.assertEqual(self.sl.ways(["A..", "A..", "..."], 1), 1)

    def test_minReorder(self):
        # 1466.Reorder Routes to Make All Paths Lead to the City Zero
        self.assertEqual(self.sl.minReorder(6, [[0, 1], [1, 3], [2, 3], [4, 0], [4, 5]]), 3)
        self.assertEqual(self.sl.minReorder(5, [[1, 0], [1, 2], [3, 2], [3, 4]]), 2)
        self.assertEqual(self.sl.minReorder(3, [[1, 0], [2, 0]]), 0)

    def test_runningSum(self):
        # 1480.Running Sum of 1d Array
        self.assertEqual(self.sl.runningSum([1, 2, 3, 4]), [1, 3, 6, 10])
        self.assertEqual(self.sl.runningSum([1, 1, 1, 1, 1]), [1, 2, 3, 4, 5])
        self.assertEqual(self.sl.runningSum([3, 1, 2, 10, 1]), [3, 4, 6, 16, 17])
        self.assertEqual(self.sl.runningSum([1]), [1])

    def test_average(self):
        # 1491.Average Salary Excluding the Minimum and Maximum Salary
        self.assertEqual(self.sl.average([4000, 3000, 1000, 2000]), 2500.00)

    def test_canMakeArithmeticProgression(self):
        # 1502.Can Make Arithmetic Progression From Sequence
        self.assertTrue(self.sl.canMakeArithmeticProgression([3, 5, 1]))
        self.assertFalse(self.sl.canMakeArithmeticProgression([1, 2, 4]))
        self.assertTrue(self.sl.canMakeArithmeticProgression([1, 2]))

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

    def test_findSmallestSetOfVertices(self):
        # 1557.Minimum Number of Vertices to Reach All Nodes
        self.assertEqual(self.sl.findSmallestSetOfVertices(6, [[0, 1], [0, 2], [2, 5], [3, 4], [4, 2]]), [0, 3])
        self.assertEqual(self.sl.findSmallestSetOfVertices(5, [[0, 1], [2, 1], [3, 1], [1, 4], [2, 4]]), [0, 2, 3])

    def test_getMaxLen(self):
        # 1567.Maximum Length of Subarray With Positive Product
        self.assertEqual(self.sl.getMaxLen([1, -2, -3, 4]), 4)
        self.assertEqual(self.sl.getMaxLen([0, 1, -2, -3, -4]), 3)
        self.assertEqual(self.sl.getMaxLen([-1, -2, -3, 0, 1]), 2)

    def test_diagonalSum(self):
        # 1572.Matrix Diagonal Sum
        self.assertEqual(self.sl.diagonalSum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 25)
        self.assertEqual(self.sl.diagonalSum([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]), 8)
        self.assertEqual(self.sl.diagonalSum([[5]]), 5)

    def test_sumOddLengthSubarrays(self):
        # 1588.Sum of All Odd Length Subarrays
        self.assertEqual(self.sl.sumOddLengthSubarrays([1, 4, 2, 5, 3]), 58)
        self.assertEqual(self.sl.sumOddLengthSubarrays([1, 2]), 3)
