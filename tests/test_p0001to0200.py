import unittest

from pysolutions import Pro0001To0200

from .tools import set_ListNode


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TestP0001To0200(unittest.TestCase):
    @property
    def sl(self):
        return Pro0001To0200()

    def test_addTwoNumbers(self):
        # 2.Add Two Numberes
        l1 = ListNode(2, next=ListNode(4, next=ListNode(3)))
        l2 = ListNode(5, next=ListNode(6, next=ListNode(4)))
        res = self.sl.addTwoNumbers(l1, l2)
        expect_res = ListNode(7, next=ListNode(0, next=ListNode(8)))
        while res:
            self.assertEqual(res.val, expect_res.val)
            res, expect_res = res.next, expect_res.next

        l1 = ListNode(0)
        l2 = ListNode(0)
        res = self.sl.addTwoNumbers(l1, l2)
        expect_res = ListNode(0)
        while res:
            self.assertEqual(res.val, expect_res.val)
            res, expect_res = res.next, expect_res.next

    def test_longestCommonPrefix(self):
        # 14.Longest Common Prefix
        strs1 = ["flower", "flow", "flight"]
        strs2 = ["dog", "racecar", "car"]
        res1 = self.sl.longestCommonPrefix(strs1)
        res2 = self.sl.longestCommonPrefix(strs2)
        expres1 = "fl"
        expres2 = ""
        self.assertEqual(res1, expres1)
        self.assertEqual(res2, expres2)

    def test_isValid(self):
        # 20.Valid Parentheses
        input1 = "()"
        input2 = "()[]{}"
        input3 = "(]"
        self.assertTrue(self.sl.isValid(input1))
        self.assertTrue(self.sl.isValid(input2))
        self.assertFalse(self.sl.isValid(input3))

    def test_mergeTwoLists(self):
        # 21.Merge Two Sorted Lists
        l11 = set_ListNode([1, 2, 4])
        l12 = set_ListNode([1, 3, 4])
        l21 = set_ListNode([])
        l22 = set_ListNode([])
        l31 = set_ListNode([])
        l32 = set_ListNode([0])
        expres1 = set_ListNode([1, 1, 2, 3, 4, 4])
        expres2 = set_ListNode([])
        expres3 = set_ListNode([0])
        res1 = self.sl.mergeTwoLists(l11, l12)
        res2 = self.sl.mergeTwoLists(l21, l22)
        res3 = self.sl.mergeTwoLists(l31, l32)
        while expres1:
            self.assertEqual(res1.val, expres1.val)
            res1, expres1 = res1.next, expres1.next
        while expres2:
            self.assertEqual(res2.val, expres2.val)
            res2, expres2 = res2.next, expres2.next
        while expres3:
            self.assertEqual(res3.val, expres3.val)
            res3, expres3 = res3.next, expres3.next
