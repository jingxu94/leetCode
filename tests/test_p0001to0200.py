import unittest

from pysolutions import Pro0001To0200


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TestP0001To0200(unittest.TestCase):
    def test_addTwoNumbers(self):
        # 2.Add Two Numberes
        l1 = ListNode(2, next=ListNode(4, next=ListNode(3)))
        l2 = ListNode(5, next=ListNode(6, next=ListNode(4)))
        res = Pro0001To0200().addTwoNumbers(l1, l2)
        expect_res = ListNode(7, next=ListNode(0, next=ListNode(8)))
        while res:
            self.assertEqual(res.val, expect_res.val)
            res, expect_res = res.next, expect_res.next

        l1 = ListNode(0)
        l2 = ListNode(0)
        res = Pro0001To0200().addTwoNumbers(l1, l2)
        expect_res = ListNode(0)
        while res:
            self.assertEqual(res.val, expect_res.val)
            res, expect_res = res.next, expect_res.next
