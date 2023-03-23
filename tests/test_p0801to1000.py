import unittest

from pysolutions import Pro0801To1000
from pysolutions.utils import create_binary_tree, create_linked_list, eq_linked_list


class TestP0801To1000(unittest.TestCase):
    @property
    def sl(self):
        return Pro0801To1000()

    def test_peakIndexInMountainArray(self):
        # 852.Peak index in a Mountain Array
        self.assertEqual(self.sl.peakIndexInMountainArray([0, 1, 0]), 1)
        self.assertEqual(self.sl.peakIndexInMountainArray([0, 2, 1, 0]), 1)
        self.assertEqual(self.sl.peakIndexInMountainArray([0, 10, 5, 2]), 1)

    def test_minEatingSpeed(self):
        # 875.Koko Eating Bananas
        piles1 = [3, 6, 7, 11]
        piles2 = [30, 11, 23, 4, 20]
        piles3 = [30, 11, 23, 4, 20]
        h1, h2, h3 = 8, 5, 6
        expres1, expres2, expres3 = 4, 30, 23
        self.assertEqual(self.sl.minEatingSpeed(piles1, h1), expres1)
        self.assertEqual(self.sl.minEatingSpeed(piles2, h2), expres2)
        self.assertEqual(self.sl.minEatingSpeed(piles3, h3), expres3)

    def test_middleNode(self):
        # 876.Middle of the Linked List
        head = create_linked_list([1, 2, 3, 4, 5])
        expected = create_linked_list([3, 4, 5])
        ans = self.sl.middleNode(head)
        self.assertTrue(eq_linked_list(ans, expected))
        head = create_linked_list([1, 2, 3, 4, 5, 6])
        expected = create_linked_list([4, 5, 6])
        ans = self.sl.middleNode(head)
        self.assertTrue(eq_linked_list(ans, expected))

    def test_sortArray(self):
        # 912.Sort an Array
        nums1 = [5, 2, 3, 1]
        nums2 = [5, 1, 1, 2, 0, 0]
        expres1 = [1, 2, 3, 5]
        expres2 = [0, 0, 1, 1, 2, 5]
        self.assertEqual(self.sl.sortArray(nums1), expres1)
        self.assertEqual(self.sl.sortArray(nums2), expres2)

    def test_isAlienSorted(self):
        # 953.Verifying an Alien Dictionary
        words = ["hello", "leetcode"]
        order = "hlabcdefgijkmnopqrstuvwxyz"
        self.assertTrue(self.sl.isAlienSorted(words, order))
        words = ["word", "world", "row"]
        order = "worldabcefghijkmnpqstuvxyz"
        self.assertFalse(self.sl.isAlienSorted(words, order))
        words = ["apple", "app"]
        order = "abcdefghijklmnopqrstuvwxyz"
        self.assertFalse(self.sl.isAlienSorted(words, order))

    def test_isCompleteTree(self):
        # 958.Check Completeness of a Binary Tree
        root1 = create_binary_tree([1, 2, 3, 4, 5, 6])
        root2 = create_binary_tree([1, 2, 3, 4, 5, None, 7])
        self.assertTrue(self.sl.isCompleteTree(root1))
        self.assertFalse(self.sl.isCompleteTree(root2))

    def test_largestPerimeter(self):
        # 976.Largest Perimeter Triangle
        self.assertEqual(self.sl.largestPerimeter([2, 1, 2]), 5)
        self.assertEqual(self.sl.largestPerimeter([1, 2, 1, 10]), 0)

    def test_sortedSquares(self):
        # 977.Squares of a Sorted Array
        self.assertListEqual(self.sl.sortedSquares([-4, -1, 0, 3, 10]), [0, 1, 9, 16, 100])
        self.assertListEqual(self.sl.sortedSquares([-7, -3, 2, 3, 11]), [4, 9, 9, 49, 121])

    def test_orangesRotting(self):
        # 994.Rotting Oranges
        self.assertEqual(self.sl.orangesRotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]), 4)
        self.assertEqual(self.sl.orangesRotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]), -1)
        self.assertEqual(self.sl.orangesRotting([[0, 2]]), 0)
