import unittest

from pysolutions import Pro1601To1800
from pysolutions.utils import create_linked_list, eq_linked_list


class TestP1601To1800(unittest.TestCase):
    @property
    def sl(self):
        return Pro1601To1800()

    def test_specialArray(self):
        # 1608.Special Array With X Elements Greater Than or Equal X
        self.assertEqual(self.sl.specialArray([3, 5]), 2)
        self.assertEqual(self.sl.specialArray([0, 0]), -1)
        self.assertEqual(self.sl.specialArray([0, 4, 3, 0, 4]), 3)

    def test_checkArithmeticSubarrays(self):
        # 1630.Arithmetic Subarrays
        nums = [-12, -9, -3, -12, -6, 15, 20, -25, -20, -15, -10]
        left = [0, 1, 6, 4, 8, 7]
        right = [4, 4, 9, 7, 9, 10]
        ans = [False, True, False, False, True, True]
        self.assertEqual(self.sl.checkArithmeticSubarrays(nums, left, right), ans)

    def test_numWays(self):
        # 1639.Number of Ways to Form a Target String Given a Dictionary
        words = ["acca", "bbbb", "caca"]
        target = "aba"
        self.assertEqual(self.sl.numWays(words, target), 6)

    def test_closeStrings(self):
        # 1657.Determine if Two Strings Are Close
        self.assertTrue(self.sl.closeStrings("abc", "bca"))
        self.assertFalse(self.sl.closeStrings("a", "aa"))
        self.assertFalse(self.sl.closeStrings("ab", "aa"))
        self.assertFalse(self.sl.closeStrings("abbzzca", "babzzcz"))
        self.assertTrue(self.sl.closeStrings("cabbba", "abbccc"))

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

    def test_interpret(self):
        # 1678.Goal Parser Interpretation
        self.assertEqual(self.sl.interpret("G()(al)"), "Goal")
        self.assertEqual(self.sl.interpret("G()()()()(al)"), "Gooooal")
        self.assertEqual(self.sl.interpret("(al)G(al)()()G"), "alGalooG")

    def test_maxOperations(self):
        # 1679.Max Number of K-Sum Pairs
        nums1, k1 = [1, 2, 3, 4], 5
        nums2, k2 = [3, 1, 3, 4, 3], 6
        nums3, k3 = [1, 1], 2
        res1, res2, res3 = 2, 1, 1
        self.assertEqual(res1, self.sl.maxOperations(nums1, k1))
        self.assertEqual(res2, self.sl.maxOperations(nums2, k2))
        self.assertEqual(res3, self.sl.maxOperations(nums3, k3))

    def test_distanceLimitedPathsExist(self):
        # 1697.Checking Existence of Edge Length Limited Paths
        n = 3
        edgeList = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]]
        queries = [[0, 1, 2], [0, 2, 5]]
        ans = [False, True]
        self.assertEqual(self.sl.distanceLimitedPathsExist(n, edgeList, queries), ans)
        n = 5
        edgeList = [[0, 1, 10], [1, 2, 5], [2, 3, 9], [3, 4, 13]]
        queries = [[0, 4, 14], [1, 4, 13]]
        ans = [True, False]
        self.assertEqual(self.sl.distanceLimitedPathsExist(n, edgeList, queries), ans)

    def test_findBall(self):
        # 1706.Where Will the Ball Fall
        grid1 = [[1, 1, 1, -1, -1], [1, 1, 1, -1, -1], [-1, -1, -1, 1, 1], [1, 1, 1, 1, -1], [-1, -1, -1, -1, -1]]
        grid2 = [[-1]]
        self.assertEqual(self.sl.findBall(grid1), [1, -1, -1, -1, -1])
        self.assertEqual(self.sl.findBall(grid2), [-1])

    def test_swapNodes(self):
        # 1721.Swapping Nodes in a Linked List
        head1 = create_linked_list([1, 2, 3, 4, 5])
        ans1 = create_linked_list([1, 4, 3, 2, 5])
        self.assertTrue(eq_linked_list(self.sl.swapNodes(head1, 2), ans1))
        head2 = create_linked_list([7, 9, 6, 6, 7, 8, 3, 0, 9, 5])
        ans2 = create_linked_list([7, 9, 6, 6, 8, 7, 3, 0, 9, 5])
        self.assertTrue(eq_linked_list(self.sl.swapNodes(head2, 5), ans2))

    def test_largestAltitude(self):
        # 1732.Find the Highest Altitude
        gain1, gain2 = [-5, 1, 5, 0, -7], [-4, -3, -2, -1, 4, 3, 2]
        self.assertEqual(self.sl.largestAltitude(gain1), 1)
        self.assertEqual(self.sl.largestAltitude(gain2), 0)

    def test_mergeAlternately(self):
        # 1768.Merge Strings Alternately
        self.assertEqual(self.sl.mergeAlternately("abc", "pqr"), "apbqcr")
        self.assertEqual(self.sl.mergeAlternately("ab", "pqrs"), "apbqrs")
        self.assertEqual(self.sl.mergeAlternately("abcd", "pq"), "apbqcd")

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
        self.assertFalse(self.sl.areAlmostEqual("aa", "ac"))

    def test_maxScore(self):
        # 1799.Maximize Score After N Operations
        nums1 = [1, 2]
        nums2 = [3, 4, 6, 8]
        nums3 = [1, 2, 3, 4, 5, 6]
        self.assertEqual(self.sl.maxScore(nums1), 1)
        self.assertEqual(self.sl.maxScore(nums2), 11)
        self.assertEqual(self.sl.maxScore(nums3), 14)
