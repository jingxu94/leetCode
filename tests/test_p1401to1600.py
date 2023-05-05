import unittest

from pysolutions import Pro1401To1600


class TestP1401To1600(unittest.TestCase):
    @property
    def sl(self):
        return Pro1401To1600()

    def test_numberOfArray(self):
        # 1416.Restore The Array
        self.assertEqual(self.sl.numberOfArrays("1000", 10000), 1)
        self.assertEqual(self.sl.numberOfArrays("1000", 10), 0)
        self.assertEqual(self.sl.numberOfArrays("1317", 2000), 8)

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

    def test_maxVowels(self):
        # 1456.Maximum Number of Vowels in a Substring of Given Length
        self.assertEqual(self.sl.maxVowels("abciiidef", 3), 3)
        self.assertEqual(self.sl.maxVowels("aeiou", 2), 2)
        self.assertEqual(self.sl.maxVowels("leetcode", 3), 2)
        self.assertEqual(self.sl.maxVowels("rhythms", 4), 0)

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

    def test_longestSubarray(self):
        # 1493.Longest Subarray of 1's After Deleting One Element
        self.assertEqual(self.sl.longestSubarray([1, 1, 0, 1]), 3)
        self.assertEqual(self.sl.longestSubarray([0, 1, 1, 1, 0, 1, 1, 0, 1]), 5)
        self.assertEqual(self.sl.longestSubarray([1, 1, 1]), 2)

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

    def test_maxNumEdgesToRemove(self):
        # 1579.Remove Max Number of Edges to Keep Graph Fully Traversable
        self.assertEqual(
            self.sl.maxNumEdgesToRemove(4, [[3, 1, 2], [3, 2, 3], [1, 1, 3], [1, 2, 4], [1, 1, 2], [2, 3, 4]]), 2
        )
        self.assertEqual(self.sl.maxNumEdgesToRemove(4, [[3, 1, 2], [3, 2, 3], [1, 1, 4], [2, 1, 4]]), 0)
        self.assertEqual(self.sl.maxNumEdgesToRemove(4, [[3, 2, 3], [1, 1, 2], [2, 3, 4]]), -1)
        edges = [
            [1, 1, 2],
            [2, 1, 3],
            [3, 2, 4],
            [3, 2, 5],
            [1, 2, 6],
            [3, 6, 7],
            [3, 7, 8],
            [3, 6, 9],
            [3, 4, 10],
            [2, 3, 11],
            [1, 5, 12],
            [3, 3, 13],
            [2, 1, 10],
            [2, 6, 11],
            [3, 5, 13],
            [1, 9, 12],
            [1, 6, 8],
            [3, 6, 13],
            [2, 1, 4],
            [1, 1, 13],
            [2, 9, 10],
            [2, 1, 6],
            [2, 10, 13],
            [2, 2, 9],
            [3, 4, 12],
            [2, 4, 7],
            [1, 1, 10],
            [1, 3, 7],
            [1, 7, 11],
            [3, 3, 12],
            [2, 4, 8],
            [3, 8, 9],
            [1, 9, 13],
            [2, 4, 10],
            [1, 6, 9],
            [3, 10, 13],
            [1, 7, 10],
            [1, 1, 11],
            [2, 4, 9],
            [3, 5, 11],
            [3, 2, 6],
            [2, 1, 5],
            [2, 5, 11],
            [2, 1, 7],
            [2, 3, 8],
            [2, 8, 9],
            [3, 4, 13],
            [3, 3, 8],
            [3, 3, 11],
            [2, 9, 11],
            [3, 1, 8],
            [2, 1, 8],
            [3, 8, 13],
            [2, 10, 11],
            [3, 1, 5],
            [1, 10, 11],
            [1, 7, 12],
            [2, 3, 5],
            [3, 1, 13],
            [2, 4, 11],
            [2, 3, 9],
            [2, 6, 9],
            [2, 1, 13],
            [3, 1, 12],
            [2, 7, 8],
            [2, 5, 6],
            [3, 1, 9],
            [1, 5, 10],
            [3, 2, 13],
            [2, 3, 6],
            [2, 2, 10],
            [3, 4, 11],
            [1, 4, 13],
            [3, 5, 10],
            [1, 4, 10],
            [1, 1, 8],
            [3, 3, 4],
            [2, 4, 6],
            [2, 7, 11],
            [2, 7, 10],
            [2, 3, 12],
            [3, 7, 11],
            [3, 9, 10],
            [2, 11, 13],
            [1, 1, 12],
            [2, 10, 12],
            [1, 7, 13],
            [1, 4, 11],
            [2, 4, 5],
            [1, 3, 10],
            [2, 12, 13],
            [3, 3, 10],
            [1, 6, 12],
            [3, 6, 10],
            [1, 3, 4],
            [2, 7, 9],
            [1, 3, 11],
            [2, 2, 8],
            [1, 2, 8],
            [1, 11, 13],
            [1, 2, 13],
            [2, 2, 6],
            [1, 4, 6],
            [1, 6, 11],
            [3, 1, 2],
            [1, 1, 3],
            [2, 11, 12],
            [3, 2, 11],
            [1, 9, 10],
            [2, 6, 12],
            [3, 1, 7],
            [1, 4, 9],
            [1, 10, 12],
            [2, 6, 13],
            [2, 2, 12],
            [2, 1, 11],
            [2, 5, 9],
            [1, 3, 8],
            [1, 7, 8],
            [1, 2, 12],
            [1, 5, 11],
            [2, 7, 12],
            [3, 1, 11],
            [3, 9, 12],
            [3, 2, 9],
            [3, 10, 11],
        ]
        self.assertEqual(self.sl.maxNumEdgesToRemove(13, edges), 114)

    def test_sumOddLengthSubarrays(self):
        # 1588.Sum of All Odd Length Subarrays
        self.assertEqual(self.sl.sumOddLengthSubarrays([1, 4, 2, 5, 3]), 58)
        self.assertEqual(self.sl.sumOddLengthSubarrays([1, 2]), 3)
