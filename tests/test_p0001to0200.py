import random
import unittest

from pysolutions import Pro0001To0200
from pysolutions.utils import ListNode, create_binary_tree, create_linked_list, eq_binary_tree, eq_linked_list


class TestP0001To0200(unittest.TestCase):
    @property
    def sl(self):
        return Pro0001To0200()

    def test_addTwoNumbers(self):
        # 2.Add Two Numberes
        l1, l2 = create_linked_list([2, 4, 3]), create_linked_list([5, 6, 4])
        ans = self.sl.addTwoNumbers(l1, l2)
        expected = ListNode(7, next=ListNode(0, next=ListNode(8)))
        self.assertTrue(eq_linked_list(ans, expected))

        l1, l2 = ListNode(0), ListNode(0)
        ans = self.sl.addTwoNumbers(l1, l2)
        expected = ListNode(0)
        self.assertTrue(eq_linked_list(ans, expected))

    def test_lengthOfLongestSubstring(self):
        # 3.Longest Substring Without Repeating Characters
        self.assertEqual(self.sl.lengthOfLongestSubstring("abcabcbb"), 3)
        self.assertEqual(self.sl.lengthOfLongestSubstring("bbbbb"), 1)
        self.assertEqual(self.sl.lengthOfLongestSubstring("pwwkew"), 3)

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

    def test_removeNthFromEnd(self):
        # 19.Remove Nth Node From of List
        self.assertTrue(
            eq_linked_list(
                self.sl.removeNthFromEnd(create_linked_list([1, 2, 3, 4, 5]), 2), create_linked_list([1, 2, 3, 5])
            )
        )
        self.assertTrue(eq_linked_list(self.sl.removeNthFromEnd(create_linked_list([1]), 1), create_linked_list([])))
        self.assertTrue(
            eq_linked_list(self.sl.removeNthFromEnd(create_linked_list([1, 2]), 1), create_linked_list([1]))
        )

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
        l11 = create_linked_list([1, 2, 4])
        l12 = create_linked_list([1, 3, 4])
        l21 = create_linked_list([])
        l22 = create_linked_list([])
        l31 = create_linked_list([])
        l32 = create_linked_list([0])
        expected1 = create_linked_list([1, 1, 2, 3, 4, 4])
        expected2 = create_linked_list([])
        expected3 = create_linked_list([0])
        ans1 = self.sl.mergeTwoLists(l11, l12)
        ans2 = self.sl.mergeTwoLists(l21, l22)
        ans3 = self.sl.mergeTwoLists(l31, l32)
        self.assertTrue(eq_linked_list(ans1, expected1))
        self.assertTrue(eq_linked_list(ans2, expected2))
        self.assertTrue(eq_linked_list(ans3, expected3))

    def test_mergeKLists(self):
        # 23.Merge k Sorted Lists
        lists = [create_linked_list([1, 4, 5]), create_linked_list([1, 3, 4]), create_linked_list([2, 6])]
        expected = create_linked_list([1, 1, 2, 3, 4, 4, 5, 6])
        ans = self.sl.mergeKLists(lists)
        self.assertTrue(eq_linked_list(ans, expected))

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

    def test_searchRange(self):
        # 34.Find First and Last Position of Element in Sorted Array
        self.assertListEqual(self.sl.searchRange([5, 7, 7, 8, 8, 10], 8), [3, 4])
        self.assertListEqual(self.sl.searchRange([5, 7, 7, 8, 8, 10], 6), [-1, -1])
        self.assertListEqual(self.sl.searchRange([], 0), [-1, -1])

    def test_searchInsert(self):
        # 35.Search Insert Position
        nums = [1, 3, 5, 6]
        self.assertEqual(self.sl.searchInsert(nums, 5), 2)
        self.assertEqual(self.sl.searchInsert(nums, 2), 1)
        self.assertEqual(self.sl.searchInsert(nums, 7), 4)

    def test_isValidSudoku(self):
        # 36.Valid Sudoku
        board1 = [
            ["5", "3", ".", ".", "7", ".", ".", ".", "."],
            ["6", ".", ".", "1", "9", "5", ".", ".", "."],
            [".", "9", "8", ".", ".", ".", ".", "6", "."],
            ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
            ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
            ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
            [".", "6", ".", ".", ".", ".", "2", "8", "."],
            [".", ".", ".", "4", "1", "9", ".", ".", "5"],
            [".", ".", ".", ".", "8", ".", ".", "7", "9"],
        ]
        board2 = [
            ["8", "3", ".", ".", "7", ".", ".", ".", "."],
            ["6", ".", ".", "1", "9", "5", ".", ".", "."],
            [".", "9", "8", ".", ".", ".", ".", "6", "."],
            ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
            ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
            ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
            [".", "6", ".", ".", ".", ".", "2", "8", "."],
            [".", ".", ".", "4", "1", "9", ".", ".", "5"],
            [".", ".", ".", ".", "8", ".", ".", "7", "9"],
        ]
        self.assertTrue(self.sl.isValidSudoku(board1))
        self.assertFalse(self.sl.isValidSudoku(board2))

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

    def test_searchMatrix(self):
        # 74.Search a 2D Matrix
        matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
        self.assertTrue(self.sl.searchMatrix(matrix, 3))
        self.assertFalse(self.sl.searchMatrix(matrix, 13))

    def test_deleteDuplicates(self):
        # 83.Remove Duplicates from Sorted List
        input1 = create_linked_list([1, 1, 2, 2, 3, 4, 5, 5, 5, 5])
        input2 = create_linked_list([1, 1, 2, 3, 3, 3, 3, 3, 4])
        input3 = create_linked_list([])
        expected1 = create_linked_list([1, 2, 3, 4, 5])
        expected2 = create_linked_list([1, 2, 3, 4])
        expected3 = create_linked_list([])
        ans1 = self.sl.deleteDuplicates(input1)
        ans2 = self.sl.deleteDuplicates(input2)
        ans3 = self.sl.deleteDuplicates(input3)
        self.assertTrue(eq_linked_list(ans1, expected1))
        self.assertTrue(eq_linked_list(ans2, expected2))
        self.assertTrue(eq_linked_list(ans3, expected3))

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

    def test_isValidBST(self):
        # 98.Validate Binary Search Tree
        self.assertTrue(self.sl.isValidBST(create_binary_tree([2, 1, 3])))
        self.assertFalse(self.sl.isValidBST(create_binary_tree([5, 1, 4, None, None, 3, 6])))

    def test_buildTree(self):
        # 106.Construct Binary Tree from Inorder and Postorder Traversal
        inorder = [9, 3, 15, 20, 7]
        postorder = [9, 15, 7, 20, 3]
        ans = self.sl.buildTree(inorder, postorder)
        expected = create_binary_tree([3, 9, 20, None, None, 15, 7])
        self.assertTrue(eq_binary_tree(ans, expected))
        inorder = [-1]
        postorder = [-1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = create_binary_tree([-1])
        self.assertTrue(eq_binary_tree(ans, expected))
        inorder = [1, 2]
        postorder = [2, 1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = create_binary_tree([1, None, 2])
        self.assertTrue(eq_binary_tree(ans, expected))
        inorder = [2, 1]
        postorder = [2, 1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = create_binary_tree([1, 2])
        self.assertTrue(eq_binary_tree(ans, expected))
        inorder = [2, 3, 1]
        postorder = [3, 2, 1]
        ans = self.sl.buildTree(inorder, postorder)
        expected = create_binary_tree([1, 2, None, None, 3])
        self.assertTrue(eq_binary_tree(ans, expected))

    def test_generate(self):
        # 118.Pascal's Triangle
        self.assertListEqual(self.sl.generate(5), [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]])
        self.assertListEqual(self.sl.generate(1), [[1]])

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
