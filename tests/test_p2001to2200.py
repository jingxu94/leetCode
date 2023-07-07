import unittest

from pysolutions import Pro2001To2200
from pysolutions.utils import create_linked_list, eq_linked_list


class TestP2001To2200(unittest.TestCase):
    @property
    def sl(self):
        return Pro2001To2200()

    def test_maxConsecutiveAnswers(self):
        # 2024.Maximize the Confusion of an Exam
        self.assertEqual(self.sl.maxConsecutiveAnswers("TTFF", 2), 4)
        self.assertEqual(self.sl.maxConsecutiveAnswers("TFFT", 1), 3)
        self.assertEqual(self.sl.maxConsecutiveAnswers("TTFTTFTT", 1), 5)

    def test_getAverages(self):
        # 2090.K Radius Subarray Averages
        self.assertEqual(self.sl.getAverages([7, 4, 3, 9, 1, 8, 5, 2, 6], 3), [-1, -1, -1, 5, 4, 4, -1, -1, -1])
        self.assertEqual(self.sl.getAverages([100000], 0), [100000])
        self.assertEqual(self.sl.getAverages([8], 100000), [-1])

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

    def test_maximumDetonation(self):
        # 2101.Detonate the Maximum Bombs
        self.assertEqual(self.sl.maximumDetonation([[2, 1, 3], [6, 1, 4]]), 2)
        self.assertEqual(self.sl.maximumDetonation([[1, 1, 5], [10, 10, 5]]), 1)
        self.assertEqual(self.sl.maximumDetonation([[1, 2, 3], [2, 3, 1], [3, 4, 2], [4, 5, 3], [5, 6, 4]]), 5)

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
