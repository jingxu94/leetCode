import unittest

from pysolutions import Pro0601To0800
from pysolutions.utils import create_binary_tree, eq_binary_tree


class TestP0601To0800(unittest.TestCase):
    @property
    def sl(self):
        return Pro0601To0800()

    def test_canPlaceFlowers(self):
        # 605.Can Place Flowers
        self.assertTrue(self.sl.canPlaceFlowers([1, 0, 0, 0, 1], 1))
        self.assertFalse(self.sl.canPlaceFlowers([1, 0, 0, 0, 1], 2))

    def test_mergeTrees(self):
        # 617.Merge Two Binary Trees
        root1 = create_binary_tree([1, 3, 2, 5])
        root2 = create_binary_tree([2, 1, 3, None, 4, None, 7])
        expected = create_binary_tree([3, 4, 5, 5, 4, None, 7])
        self.assertTrue(eq_binary_tree(self.sl.mergeTrees(root1, root2), expected))
        root1 = create_binary_tree([1])
        root2 = create_binary_tree([1, 2])
        expected = create_binary_tree([2, 2])
        self.assertTrue(eq_binary_tree(self.sl.mergeTrees(root1, root2), expected))
        root1 = create_binary_tree([3, 4, 5, 1, 2])
        root2 = create_binary_tree([4, 1, 2])
        expected = create_binary_tree([7, 5, 7, 1, 2])
        self.assertTrue(eq_binary_tree(self.sl.mergeTrees(root1, root2), expected))
        self.assertTrue(eq_binary_tree(self.sl.mergeTrees(root1, None), root1))
        self.assertTrue(eq_binary_tree(self.sl.mergeTrees(None, root1), root1))
        self.assertTrue(eq_binary_tree(self.sl.mergeTrees(None, None), None))

    def test_leastInterval(self):
        # 621.Task Scheduler
        self.assertEqual(self.sl.leastInterval(["A", "A", "A", "B", "B", "B"], 2), 8)
        self.assertEqual(self.sl.leastInterval(["A", "A", "A", "B", "B", "B"], 0), 6)
        self.assertEqual(self.sl.leastInterval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2), 16)

    def test_judgeSquareSum(self):
        # 633.Sum of Square Numbers
        self.assertTrue(self.sl.judgeSquareSum(5))
        self.assertTrue(self.sl.judgeSquareSum(25))
        self.assertTrue(self.sl.judgeSquareSum(26))
        self.assertFalse(self.sl.judgeSquareSum(3))
        self.assertFalse(self.sl.judgeSquareSum(24))

    def test_predictPartyVictory(self):
        # 649.Dota2 Senate
        self.assertEqual(self.sl.predictPartyVictory("RD"), "Radiant")
        self.assertEqual(self.sl.predictPartyVictory("RDD"), "Dire")

    def test_findTarget(self):
        # 653.Two Sum IV - Input is a BST
        root = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        self.assertTrue(self.sl.findTarget(root, 9))
        root = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        self.assertFalse(self.sl.findTarget(root, 28))
        self.assertFalse(self.sl.findTarget(create_binary_tree([]), 28))

    def test_widthOfBinaryTree(self):
        # 662.Maximum Width of Binary Tree
        root = create_binary_tree([1, 3, 2, 5, 3, None, 9])
        self.assertEqual(self.sl.widthOfBinaryTree(root), 4)
        root = create_binary_tree([1, 3, None, 5, 3])
        self.assertEqual(self.sl.widthOfBinaryTree(root), 2)
        root = create_binary_tree([1, 3, 2, 5])
        self.assertEqual(self.sl.widthOfBinaryTree(root), 2)
        self.assertEqual(self.sl.widthOfBinaryTree(None), 0)

    def test_findNumberOfLIS(self):
        # 673.Number of Longest Increasing Subsequence
        self.assertEqual(self.sl.findNumberOfLIS([1, 3, 5, 4, 7]), 2)
        self.assertEqual(self.sl.findNumberOfLIS([2, 2, 2, 2, 2]), 5)
        self.assertEqual(self.sl.findNumberOfLIS([1, 2, 3, 1, 2, 3, 1, 2, 3]), 10)
        self.assertEqual(self.sl.findNumberOfLIS([]), 0)

    def test_topKFrequent(self):
        # 692.Top K Frequent Words
        words = ["i", "love", "leetcode", "i", "love", "coding"]
        self.assertEqual(self.sl.topKFrequent(words, 2), ["i", "love"])
        words = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
        self.assertEqual(self.sl.topKFrequent(words, 4), ["the", "is", "sunny", "day"])

    def test_maxAreaOfIsland(self):
        # 695.Max Area of Island
        grid = [
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        ]
        self.assertEqual(self.sl.maxAreaOfIsland(grid), 6)
        self.assertEqual(self.sl.maxAreaOfIsland([[0, 0, 0, 0, 0, 0, 0, 0]]), 0)

    def test_searchBST(self):
        # 700.Search in a Binary Tree
        example1 = eq_binary_tree(
            create_binary_tree([2, 1, 3]), self.sl.searchBST(create_binary_tree([4, 2, 7, 1, 3]), 2)
        )
        self.assertTrue(example1)
        example2 = eq_binary_tree(create_binary_tree([]), self.sl.searchBST(create_binary_tree([4, 2, 7, 1, 3]), 5))
        self.assertTrue(example2)

    def test_insertIntoBST(self):
        # 701.Insert into a Binary Search Tree
        root = create_binary_tree([4, 2, 7, 1, 3])
        expected = create_binary_tree([4, 2, 7, 1, 3, 5])
        self.assertTrue(eq_binary_tree(expected, self.sl.insertIntoBST(root, 5)))
        root = create_binary_tree([40, 20, 60, 10, 30, 50, 70])
        expected = create_binary_tree([40, 20, 60, 10, 30, 50, 70, None, None, 25])
        self.assertTrue(eq_binary_tree(expected, self.sl.insertIntoBST(root, 25)))
        root = create_binary_tree([4, 2, 7, 1, 3, None, None, None, None, None, None])
        expected = create_binary_tree([4, 2, 7, 1, 3, 5])
        self.assertTrue(eq_binary_tree(expected, self.sl.insertIntoBST(root, 5)))

    def test_search(self):
        # 704.Binary Search
        self.assertEqual(self.sl.search([-1, 0, 3, 5, 9, 12], 9), 4)
        self.assertEqual(self.sl.search([-1, 0, 3, 5, 9, 12], 2), -1)

    def test_toLowerCase(self):
        # 709.To Lower Case
        self.assertEqual(self.sl.toLowerCase("Hello"), "hello")
        self.assertEqual(self.sl.toLowerCase("here"), "here")
        self.assertEqual(self.sl.toLowerCase("LOVELY"), "lovely")

    def test_numSubarrayProductLessThanK(self):
        # 713.Subarray Product Less Than K
        self.assertEqual(self.sl.numSubarrayProductLessThanK([10, 5, 2, 6], 100), 8)
        self.assertEqual(self.sl.numSubarrayProductLessThanK([1, 2, 3], 0), 0)

    def test_maxProfit(self):
        # 714.Best Time to Buy ans Sell Stock with Transaction Fee
        self.assertEqual(self.sl.maxProfit([1, 3, 2, 8, 4, 9], 2), 8)
        self.assertEqual(self.sl.maxProfit([1, 3, 7, 5, 10, 3], 3), 6)

    def test_pivotIndex(self):
        # 724.Find Pivot Index
        self.assertEqual(self.sl.pivotIndex([1, 7, 3, 6, 5, 6]), 3)
        self.assertEqual(self.sl.pivotIndex([1, 2, 3]), -1)
        self.assertEqual(self.sl.pivotIndex([2, 1, -1]), 0)

    def test_floodFill(self):
        # 733.Flood Fill
        image, sr, sc, color = [[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2
        expected = [[2, 2, 2], [2, 2, 0], [2, 0, 1]]
        self.assertListEqual(self.sl.floodFill(image, sr, sc, color), expected)
        image, sr, sc, color = [[0, 0, 0], [0, 0, 0]], 0, 0, 0
        expected = [[0, 0, 0], [0, 0, 0]]
        self.assertListEqual(self.sl.floodFill(image, sr, sc, color), expected)

    def test_dailyTemperatures(self):
        # 739.Daily Temperatures
        self.assertEqual(self.sl.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])
        self.assertEqual(self.sl.dailyTemperatures([30, 40, 50, 60]), [1, 1, 1, 0])
        self.assertEqual(self.sl.dailyTemperatures([30, 60, 90]), [1, 1, 0])

    def test_deleteAndEarn(self):
        # 740.Delete and Earn
        self.assertEqual(self.sl.deleteAndEarn([3, 4, 2]), 6)
        self.assertEqual(self.sl.deleteAndEarn([2, 2, 3, 3, 3, 4]), 9)

    def test_nextGreatestLetter(self):
        # 744.Find Smallest Letter Greater Than Target
        self.assertEqual(self.sl.nextGreatestLetter(["c", "f", "j"], "a"), "c")
        self.assertEqual(self.sl.nextGreatestLetter(["c", "f", "j"], "c"), "f")
        self.assertEqual(self.sl.nextGreatestLetter(["x", "x", "y", "y"], "z"), "x")

    def test_minCostClimbingStairs(self):
        # 746.Min Cost Climbing Stairs
        self.assertEqual(self.sl.minCostClimbingStairs([10, 15, 20]), 15)
        self.assertEqual(self.sl.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]), 6)
        self.assertEqual(self.sl.minCostClimbingStairs([10]), 0)

    def test_partitionLabels(self):
        # 763.Partition Labels
        self.assertEqual(self.sl.partitionLabels("ababcbacadefegdehijhklij"), [9, 7, 8])
        self.assertEqual(self.sl.partitionLabels("eccbbbbdec"), [10])

    def test_letterCasePermutation(self):
        # 784.Letter Case Permutation
        self.assertEqual(set(self.sl.letterCasePermutation("a1b2")), set(["a1b2", "a1B2", "A1b2", "A1B2"]))
        self.assertEqual(set(self.sl.letterCasePermutation("3z4")), set(["3z4", "3Z4"]))
        self.assertCountEqual(self.sl.letterCasePermutation("a1b2"), ["a1b2", "a1B2", "A1b2", "A1B2"])
        self.assertCountEqual(self.sl.letterCasePermutation("3z4"), ["3z4", "3Z4"])
        self.assertCountEqual(self.sl.letterCasePermutation_v2("a1b2"), ["a1b2", "a1B2", "A1b2", "A1B2"])
        self.assertCountEqual(self.sl.letterCasePermutation_v2("3z4"), ["3z4", "3Z4"])

    def test_allPathsSourceTarget(self):
        # 797. All Path From Source to Target
        self.assertEqual(self.sl.allPathsSourceTarget([[1, 2], [3], [3], []]), [[0, 2, 3], [0, 1, 3]])
        # self.assertEqual(
        #     self.sl.allPathsSourceTarget([[4, 3, 1], [3, 2, 4], [3], [4], []]),
        #     [[0, 4], [0, 3, 4], [0, 1, 3, 4], [0, 1, 2, 3, 4], [0, 1, 4]],
        # )
