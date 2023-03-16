import unittest

from pysolutions import Pro0201To0400


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TestP0201To0400(unittest.TestCase):
    @property
    def sl(self):
        return Pro0201To0400()

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
