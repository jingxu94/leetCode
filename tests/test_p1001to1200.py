import unittest

from pysolutions import Pro1001To1200
from pysolutions.utils import create_binary_tree


class TestP1001To1200(unittest.TestCase):
    @property
    def sl(self):
        return Pro1001To1200()

    def test_commonChars(self):
        # 1002.Find Common Characters
        words1 = ["bella", "label", "roller"]
        words2 = ["cool", "lock", "cook"]
        self.assertListEqual(self.sl.commonChars(words1), ["e", "l", "l"])
        self.assertListEqual(self.sl.commonChars(words2), ["c", "o"])
        self.assertListEqual(self.sl.commonChars(["words"]), ["w", "o", "r", "d", "s"])

    def test_longestOnes(self):
        # 1004.Max Consecutive Ones III
        self.assertEqual(self.sl.longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], 2), 6)
        self.assertEqual(self.sl.longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], 0), 4)

    def test_maxScoreSightseeingPair(self):
        # 1014.Best Sightseeing Pair
        self.assertEqual(self.sl.maxScoreSightseeingPair([8, 1, 5, 2, 6]), 11)
        self.assertEqual(self.sl.maxScoreSightseeingPair([1, 2]), 2)

    def test_numEnclaves(self):
        # 1020.Number of Enclaves
        self.assertEqual(self.sl.numEnclaves([[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]), 3)

    def test_maxUncrossedLines(self):
        # 1035.Uncrossed Lines
        self.assertEqual(self.sl.maxUncrossedLines([1, 4, 2], [1, 2, 4]), 2)
        self.assertEqual(self.sl.maxUncrossedLines([2, 5, 1, 2, 5], [10, 5, 2, 1, 5, 2]), 3)
        self.assertEqual(self.sl.maxUncrossedLines([1, 3, 7, 1, 7, 5], [1, 9, 2, 5, 1]), 2)

    def test_isRobotBounded(self):
        # 1041.Robot Bounded In Circle
        self.assertTrue(self.sl.isRobotBounded("GGLLGG"))
        self.assertFalse(self.sl.isRobotBounded("GG"))
        self.assertTrue(self.sl.isRobotBounded("GL"))
        self.assertTrue(self.sl.isRobotBounded("RLLGGLRGLGLLLGRLRLRLRRRRLRLGRLLLGGL"))

    def test_lastStoneWeight(self):
        # 1046.Last Stone Weight
        self.assertEqual(self.sl.lastStoneWeight([2, 7, 4, 1, 8, 1]), 1)
        self.assertEqual(self.sl.lastStoneWeight([1]), 1)
        self.assertEqual(self.sl.lastStoneWeight([1, 1, 2, 2]), 0)

    def test_gcdOfStrings(self):
        # 1071.Greatest Common Divisor of Strings
        self.assertEqual(self.sl.gcdOfStrings("ABCABC", "ABC"), "ABC")
        self.assertEqual(self.sl.gcdOfStrings("ABABAB", "ABAB"), "AB")
        self.assertEqual(self.sl.gcdOfStrings("LEET", "CODE"), "")

    def test_shortestPathBinaryMatrix(self):
        # 1091.Shortest Path in Binary Matrix
        self.assertEqual(self.sl.shortestPathBinaryMatrix([[0, 1], [1, 0]]), 2)
        self.assertEqual(self.sl.shortestPathBinaryMatrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]), 4)
        self.assertEqual(self.sl.shortestPathBinaryMatrix([[1, 1], [1, 0]]), -1)
        self.assertEqual(self.sl.shortestPathBinaryMatrix([[0, 1], [1, 1]]), -1)

    def test_tribonacci(self):
        # 1137.N-th Tribonacci Number
        self.assertEqual(self.sl.tribonacci(1), 1)
        self.assertEqual(self.sl.tribonacci(2), 1)
        self.assertEqual(self.sl.tribonacci(4), 4)
        self.assertEqual(self.sl.tribonacci(25), 1389537)

    def test_stoneGameII(self):
        # 1140.Stone Game II
        self.assertEqual(self.sl.stoneGameII([2, 7, 9, 4, 4]), 10)
        self.assertEqual(self.sl.stoneGameII([1, 2, 3, 4, 5, 100]), 104)
        piles = [
            3111,
            4303,
            2722,
            2183,
            6351,
            5227,
            8964,
            7167,
            9286,
            6626,
            2347,
            1465,
            5201,
            7240,
            5463,
            8523,
            8163,
            9391,
            8616,
            5063,
            7837,
            7050,
            1246,
            9579,
            7744,
            6932,
            7704,
            9841,
            6163,
            4829,
            7324,
            6006,
            4689,
            8781,
            621,
        ]
        self.assertEqual(self.sl.stoneGameII(piles), 112766)

    def test_longestCommonSubsequence(self):
        # 1143.Longest Common Subsequence
        self.assertEqual(self.sl.longestCommonSubsequence("abcde", "ace"), 3)
        self.assertEqual(self.sl.longestCommonSubsequence("abc", "abc"), 3)
        self.assertEqual(self.sl.longestCommonSubsequence("abc", "def"), 0)

    def test_maxLevelSum(self):
        # 1161.Maximum Level Sum of a Binary Tree
        root = create_binary_tree([1, 7, 0, 7, -8, None, None])
        self.assertEqual(self.sl.maxLevelSum(root), 2)
        root = create_binary_tree([989, None, 10250, 98693, -89388, None, None, None, -32127])
        self.assertEqual(self.sl.maxLevelSum(root), 2)
