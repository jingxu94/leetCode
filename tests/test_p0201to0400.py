import unittest
from typing import Optional

from pysolutions import Pro0201To0400

from .tools import ListNode, set_ListNode


class TestP0201To0400(unittest.TestCase):
    @property
    def sl(self):
        return Pro0201To0400()

    def _eq_ListNode(self, ans: Optional[ListNode], expected: Optional[ListNode]):
        while ans and expected:
            self.assertEqual(ans.val, expected.val)
            ans, expected = ans.next, expected.next
        if ans or expected:
            raise ValueError("ListNode with different length!")

    def test_isIsomorphic(self):
        # 205.Isomorphic Strings
        self.assertTrue(self.sl.isIsomorphic("egg", "add"))
        self.assertFalse(self.sl.isIsomorphic("foo", "bar"))
        self.assertTrue(self.sl.isIsomorphic("paper", "title"))
        self.assertFalse(self.sl.isIsomorphic("badc", "baba"))

    def test_reverseList(self):
        # 206.Reverse Linked List
        self._eq_ListNode(self.sl.reverseList(set_ListNode([1, 2, 3, 4, 5])), set_ListNode([5, 4, 3, 2, 1]))
        self._eq_ListNode(self.sl.reverseList(set_ListNode([1, 2])), set_ListNode([2, 1]))
        self._eq_ListNode(self.sl.reverseList(set_ListNode([])), set_ListNode([]))

    def test_containDuplicate(self):
        # 217.Contains Duplicate
        self.assertTrue(self.sl.containsDuplicate([1, 2, 3, 1]))
        self.assertTrue(self.sl.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))
        self.assertFalse(self.sl.containsDuplicate([1, 2, 3, 4]))

    def test_moveZeroes(self):
        # 283.Move Zeroes
        self.assertListEqual(self.sl.moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
        self.assertEqual(self.sl.moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
        self.assertListEqual(self.sl.moveZeroes([0]), [0])
        self.assertEqual(self.sl.moveZeroes([0]), [0])

    def test_intersect(self):
        # 350.Intersection of Two Arrays 2
        self.assertEqual(self.sl.intersect([1, 2, 2, 1], [2, 2]), [2, 2])
        self.assertEqual(self.sl.intersect([4, 9, 5], [9, 4, 9, 8, 4]), [4, 9])

    def test_isPerfectSquare(self):
        # 367.Valid Perfect Square
        self.assertTrue(self.sl.isPerfectSquare(16))
        self.assertFalse(self.sl.isPerfectSquare(14))

    def test_isSubsequence(self):
        # 392.Is Subsequence
        self.assertTrue(self.sl.isSubsequence("abc", "ahbgdc"))
        self.assertFalse(self.sl.isSubsequence("axc", "ahbgdc"))
