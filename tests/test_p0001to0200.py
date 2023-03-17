import random
import unittest
from collections import deque
from typing import Optional

from pysolutions import Pro0001To0200

from .tools import ListNode, TreeNode, set_ListNode, set_TreeNode


class TestP0001To0200(unittest.TestCase):
    @property
    def sl(self):
        return Pro0001To0200()

    def _eq_ListNode(self, ans: Optional[ListNode], expected: Optional[ListNode]):
        while ans and expected:
            self.assertEqual(ans.val, expected.val)
            ans, expected = ans.next, expected.next
        if ans or expected:
            raise ValueError("ListNode with different length!")

    def _isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # 100.Same Tree
        def check(p, q):
            # if both are None
            if not p and not q:
                return True
            # one of p and q is None
            if not q or not p:
                return False
            if p.val != q.val:
                return False
            return True

        deq = deque(
            [
                (p, q),
            ]
        )
        while deq:
            p, q = deq.popleft()
            if not check(p, q):
                return False

            if p:
                deq.append((p.left, q.left))
                deq.append((p.right, q.right))

        return True

    def test_addTwoNumbers(self):
        # 2.Add Two Numberes
        l1, l2 = set_ListNode([2, 4, 3]), set_ListNode([5, 6, 4])
        ans = self.sl.addTwoNumbers(l1, l2)
        expected = ListNode(7, next=ListNode(0, next=ListNode(8)))
        self._eq_ListNode(ans, expected)

        l1, l2 = ListNode(0), ListNode(0)
        ans = self.sl.addTwoNumbers(l1, l2)
        expected = ListNode(0)
        self._eq_ListNode(ans, expected)

    def test_longestCommonPrefix(self):
        # 14.Longest Common Prefix
        strs1 = ["flower", "flow", "flight"]
        strs2 = ["dog", "racecar", "car"]
        res1 = self.sl.longestCommonPrefix(strs1)
        res2 = self.sl.longestCommonPrefix(strs2)
        expected1 = "fl"
        expected2 = ""
        self.assertEqual(res1, expected1)
        self.assertEqual(res2, expected2)

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
        expected1 = set_ListNode([1, 1, 2, 3, 4, 4])
        expected2 = set_ListNode([])
        expected3 = set_ListNode([0])
        ans1 = self.sl.mergeTwoLists(l11, l12)
        ans2 = self.sl.mergeTwoLists(l21, l22)
        ans3 = self.sl.mergeTwoLists(l31, l32)
        self._eq_ListNode(ans1, expected1)
        self._eq_ListNode(ans2, expected2)
        self._eq_ListNode(ans3, expected3)

    def test_mergeKLists(self):
        # 23.Merge k Sorted Lists
        lists = [set_ListNode([1, 4, 5]), set_ListNode([1, 3, 4]), set_ListNode([2, 6])]
        expected = set_ListNode([1, 1, 2, 3, 4, 4, 5, 6])
        ans = self.sl.mergeKLists(lists)
        self._eq_ListNode(ans, expected)

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
        expected1 = [1, 2]
        expected2 = [0, 1, 2, 3, 4]
        self.assertEqual(self.sl.removeDuplicates(nums1), expected1)
        self.assertEqual(self.sl.removeDuplicates(nums2), expected2)

    def test_strStr(self):
        # 28.Find the Index of the First Occurrence in a String
        haystack1, needle1 = "sadbutsad", "sad"
        expected1 = 0
        haystack2, needle2 = "leetcode", "leeto"
        expected2 = -1
        self.assertEqual(self.sl.strStr(haystack1, needle1), expected1)
        self.assertEqual(self.sl.strStr(haystack2, needle2), expected2)

    def test_searchInsert(self):
        # 35.Search Insert Position
        nums = [1, 3, 5, 6]
        self.assertEqual(self.sl.searchInsert(nums, 5), 2)
        self.assertEqual(self.sl.searchInsert(nums, 2), 1)
        self.assertEqual(self.sl.searchInsert(nums, 7), 4)

    def test_maxSubArray(self):
        # 53.Maximum Subarray
        self.assertEqual(self.sl.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
        self.assertEqual(self.sl.maxSubArray([1]), 1)
        self.assertEqual(self.sl.maxSubArray([5, 4, -1, 7, 8]), 23)

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
        expected1 = set_ListNode([1, 2, 3, 4, 5])
        expected2 = set_ListNode([1, 2, 3, 4])
        expected3 = set_ListNode([])
        ans1 = self.sl.deleteDuplicates(input1)
        ans2 = self.sl.deleteDuplicates(input2)
        ans3 = self.sl.deleteDuplicates(input3)
        self._eq_ListNode(ans1, expected1)
        self._eq_ListNode(ans2, expected2)
        self._eq_ListNode(ans3, expected3)

    def test_merge(self):
        # 88.Merge Sorted Array
        nums1, m, nums2, n = [1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3
        self.sl.merge(nums1, m, nums2, n)
        self.assertEqual(nums1, [1, 2, 2, 3, 5, 6])
        nums1, m, nums2, n = [1], 1, [], 0
        self.sl.merge(nums1, m, nums2, n)
        self.assertEqual(nums1, [1])
        nums1, m, nums2, n = [0], 0, [1], 1
        self.sl.merge(nums1, m, nums2, n)
        self.assertEqual(nums1, [1])

    @unittest.skip("FIXME: There is a bug in set TreeNode when None in elements")
    def test_buildTree(self):
        # 106.Construct Binary Tree from Inorder and Postorder Traversal
        inorder = [9, 3, 15, 20, 7]
        postorder = [9, 15, 7, 20, 3]
        ans = self.sl.buildTree(inorder, postorder)
        expected = set_TreeNode([3, 9, 20, None, None, 15, 7])
        self.assertTrue(self._isSameTree(ans, expected))
        inorder = [-1]
        postorder = [-1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = set_TreeNode([-1])
        self.assertTrue(self._isSameTree(ans, expected))
        inorder = [1, 2]
        postorder = [2, 1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = set_TreeNode([1, None, 2])
        self.assertTrue(self._isSameTree(ans, expected))
        inorder = [2, 1]
        postorder = [2, 1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = set_TreeNode([1, 2])
        self.assertTrue(self._isSameTree(ans, expected))
        inorder = [2, 3, 1]
        postorder = [3, 2, 1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = set_TreeNode([1, 2, None, None, 3])
        self.assertTrue(self._isSameTree(ans, expected))

    def test_maxProfit(self):
        # 121.Best Time to Buy and Sell Stock
        self.assertEqual(self.sl.maxProfit([7, 1, 5, 3, 6, 4]), 5)
        self.assertEqual(self.sl.maxProfit([7, 6, 4, 3, 1]), 0)

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

    def test_reverseWords(self):
        # 151.Reverse Words in a String
        s1 = "the sky is blue"
        s2 = "  hello world  "
        s3 = "a good   example"
        expected1 = "blue is sky the"
        expected2 = "world hello"
        expected3 = "example good a"
        self.assertEqual(self.sl.reverseWords(s1), expected1)
        self.assertEqual(self.sl.reverseWords(s2), expected2)
        self.assertEqual(self.sl.reverseWords(s3), expected3)

    def test_twoSum(self):
        # 167.Two Sum 2 - Input Array Is Sorted
        self.assertEqual(self.sl.twoSum([2, 7, 11, 15], 9), [1, 2])
        self.assertEqual(self.sl.twoSum([2, 3, 4], 6), [1, 3])
        self.assertEqual(self.sl.twoSum([-1, 0], -1), [1, 2])
        self.assertEqual(self.sl.twoSum([0, 0, 3, 4], 0), [1, 2])

    def test_hammingWeight(self):
        # 191.Number of 1 Bits
        input1 = 0b00000000000000000000000000001011
        input2 = 0b00000000000000000000000010000000
        input3 = 0b11111111111111111111111111111101
        self.assertEqual(self.sl.hammingWeight(input1), 3)
        self.assertEqual(self.sl.hammingWeight(input2), 1)
        self.assertEqual(self.sl.hammingWeight(input3), 31)
