import random
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

    def test_detectCycle(self):
        # 142.Linked List Cycle II
        input = ListNode()
        curr = input
        curr.next = ListNode(3)
        curr = curr.next
        curr.next = ListNode(2)
        curr = curr.next
        cycle = curr
        curr.next = ListNode(0)
        curr = curr.next
        curr.next = ListNode(-4)
        curr = curr.next
        curr.next = cycle
        self.assertEqual(self.sl.detectCycle(input.next).val, 2)

    def test_removeDuplicates(self):
        # 26.Remove Duplicates from Sorted Array
        # === Custom Judege ===
        # int[] nums = [...]; // Input array
        # int[] expectedNums = [...]; // The expected answer with correct length
        # int k = removeDuplicates(nums); // Calls your implementation
        # assert k == expectedNums.length;
        # for (int i = 0; i < k; i++) {
        #     assert nums[i] == expectedNums[i];
        # }
        # =====================
        nums1 = [1, 1, 2]
        nums2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        expres1 = [1, 2]
        expres2 = [0, 1, 2, 3, 4]
        self.assertEqual(self.sl.removeDuplicates(nums1), expres1)
        self.assertEqual(self.sl.removeDuplicates(nums2), expres2)

    def test_strStr(self):
        # 28.Find the Index of the First Occurrence in a String
        haystack1, needle1 = "sadbutsad", "sad"
        expres1 = 0
        haystack2, needle2 = "leetcode", "leeto"
        expres2 = -1
        self.assertEqual(self.sl.strStr(haystack1, needle1), expres1)
        self.assertEqual(self.sl.strStr(haystack2, needle2), expres2)

    def test_searchInsert(self):
        # 35.Search Insert Position
        nums = [1, 3, 5, 6]
        self.assertEqual(self.sl.searchInsert(nums, 5), 2)
        self.assertEqual(self.sl.searchInsert(nums, 2), 1)
        self.assertEqual(self.sl.searchInsert(nums, 7), 4)

    def test_plusOne(self):
        # 66.Plus One
        self.assertEqual(self.sl.plusOne([1, 2, 3]), [1, 2, 4])
        self.assertEqual(self.sl.plusOne([4, 3, 2, 1]), [4, 3, 2, 2])
        self.assertEqual(self.sl.plusOne([9]), [1, 0])

    def test_addBinary(self):
        # 67.Add Binary
        self.assertEqual(self.sl.addBinary("11", "1"), "100")
        self.assertEqual(self.sl.addBinary("1010", "1011"), "10101")

    def test_mySqrt(self):
        # 69.Sqrt(x)
        for _ in range(10):
            num = random.randint(0, 2**31 - 1)
            self.assertEqual(self.sl.mySqrt(num), int(num**0.5))

    def test_climbStairs(self):
        # 70.Climbing Stairs
        inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        outputs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        for input, output in zip(inputs, outputs):
            self.assertEqual(self.sl.climbStairs(input), output)

    def test_deleteDuplicates(self):
        # 83.Remove Duplicates from Sorted List
        input1 = set_ListNode([1, 1, 2, 2, 3, 4, 5, 5, 5, 5])
        input2 = set_ListNode([1, 1, 2, 3, 3, 3, 3, 3, 4])
        input3 = set_ListNode([])
        expres1 = set_ListNode([1, 2, 3, 4, 5])
        expres2 = set_ListNode([1, 2, 3, 4])
        expres3 = set_ListNode([])
        res1 = self.sl.deleteDuplicates(input1)
        res2 = self.sl.deleteDuplicates(input2)
        res3 = self.sl.deleteDuplicates(input3)
        while expres1:
            self.assertEqual(res1.val, expres1.val)
            res1, expres1 = res1.next, expres1.next
        while expres2:
            self.assertEqual(res2.val, expres2.val)
            res2, expres2 = res2.next, expres2.next
        while expres3:
            self.assertEqual(res3.val, expres3.val)
            res3, expres3 = res3.next, expres3.next
