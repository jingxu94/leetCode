import unittest

from pysolutions import Pro2001To2200
from pysolutions.utils import create_linked_list, eq_linked_list


class TestP2001To2200(unittest.TestCase):
    @property
    def sl(self):
        return Pro2001To2200()

    def test_deleteMiddle(self):
        # 2095.Delete the Middle Node of a Linked List
        self.assertEqual(self.sl.deleteMiddle(None), None)
        head = create_linked_list([1, 3, 4, 7, 1, 2, 6])
        expected = create_linked_list([1, 3, 4, 1, 2, 6])
        self.assertTrue(eq_linked_list(self.sl.deleteMiddle(head), expected))
        head = create_linked_list([1, 2, 3, 4])
        expected = create_linked_list([1, 2, 4])
        self.assertTrue(eq_linked_list(self.sl.deleteMiddle(head), expected))
        head = create_linked_list([2, 1])
        expected = create_linked_list([2])
        self.assertTrue(eq_linked_list(self.sl.deleteMiddle(head), expected))

    def test_pairSum(self):
        # 2130.Maximum Twin Sum of a Linked List
        head = create_linked_list([5, 4, 2, 1])
        self.assertEqual(self.sl.pairSum(head), 6)
        head = create_linked_list([4, 2, 2, 3])
        self.assertEqual(self.sl.pairSum(head), 7)
        head = create_linked_list([1, 100000])
        self.assertEqual(self.sl.pairSum(head), 100001)
        self.assertEqual(self.sl.pairSum(None), 0)

    def test_longestPalindrome(self):
        # 2131.Longest Palindrome by Concatenating Two Letter Words
        self.assertEqual(self.sl.longestPalindrome(["lc", "cl", "gg"]), 6)
        self.assertEqual(self.sl.longestPalindrome(["ab", "ty", "yt", "lc", "cl", "ab"]), 8)
        self.assertEqual(self.sl.longestPalindrome(["cc", "ll", "xx"]), 2)

    def test_mostPoints(self):
        # 2140.Solving Questions With Brainpower
        self.assertEqual(self.sl.mostPoints([[3, 2], [4, 3], [4, 4], [2, 5]]), 5)
        self.assertEqual(self.sl.mostPoints([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]), 7)

    def test_minimumTime(self):
        # 2187.Minimum Time to Complete Trips
        self.assertEqual(self.sl.minimumTime([1, 2, 3], 5), 3)
        self.assertEqual(self.sl.minimumTime([2], 1), 2)
