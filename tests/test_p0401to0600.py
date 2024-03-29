import unittest

from pysolutions import Pro0401To0600
from pysolutions.utils import create_binary_tree, create_linked_list, eq_binary_tree, eq_linked_list


class TestP0401To0600(unittest.TestCase):
    @property
    def sl(self):
        return Pro0401To0600()

    def test_sumOfLeftLeaves(self):
        # 404.Sum of Left Leaves
        self.assertEqual(self.sl.sumOfLeftLeaves(create_binary_tree([])), 0)
        self.assertEqual(self.sl.sumOfLeftLeaves(create_binary_tree([3, 9, 20, None, None, 15, 7])), 24)
        self.assertEqual(self.sl.sumOfLeftLeaves(create_binary_tree([1])), 0)

    def test_longestPalindrom(self):
        # 409.Longest Palindrome
        self.assertEqual(self.sl.longestPalindrome("abccccdd"), 7)
        self.assertEqual(self.sl.longestPalindrome("a"), 1)
        self.assertEqual(self.sl.longestPalindrome("aa"), 2)

    def test_fizzBuzz(self):
        # 412.Fizz Buzz
        self.assertEqual(self.sl.fizzBuzz(3), ["1", "2", "Fizz"])
        self.assertEqual(self.sl.fizzBuzz(5), ["1", "2", "Fizz", "4", "Buzz"])
        self.assertEqual(
            self.sl.fizzBuzz(15),
            ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"],
        )

    def test_numberOfArithmeticSlices(self):
        # 413.Arthmetic Slices
        self.assertEqual(self.sl.numberOfArithmeticSlices([1, 2, 3, 4]), 3)
        self.assertEqual(self.sl.numberOfArithmeticSlices([1]), 0)
        self.assertEqual(self.sl.numberOfArithmeticSlices([1, 2, 4, 9]), 0)

    def test_addString(self):
        # 415.Add Strings
        self.assertEqual(self.sl.addStrings("11", "123"), "134")
        self.assertEqual(self.sl.addStrings("456", "77"), "533")
        self.assertEqual(self.sl.addStrings("0", "0"), "0")

    def test_canPartition(self):
        # 416.Partition Equal Subset Sum
        self.assertTrue(self.sl.canPartition([1, 5, 11, 5]))
        self.assertFalse(self.sl.canPartition([1, 2, 3, 5]))

    def test_characterReplacement(self):
        # 424.Longest Repeating Character Replacement
        self.assertEqual(self.sl.characterReplacement("ABAB", 2), 4)
        self.assertEqual(self.sl.characterReplacement("AABABBA", 1), 4)
        self.assertEqual(self.sl.characterReplacement("AABABBA", 2), 5)

    def test_eraseOverlapIntervals(self):
        # 435.Non-overlapping Intervals
        self.assertEqual(self.sl.eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]), 1)
        self.assertEqual(self.sl.eraseOverlapIntervals([[1, 2], [1, 2], [1, 2]]), 2)
        self.assertEqual(self.sl.eraseOverlapIntervals([[1, 2], [2, 3]]), 0)
        self.assertEqual(self.sl.eraseOverlapIntervals_v2([[1, 2], [2, 3], [3, 4], [1, 3]]), 1)
        self.assertEqual(self.sl.eraseOverlapIntervals_v2([[1, 2], [1, 2], [1, 2]]), 2)
        self.assertEqual(self.sl.eraseOverlapIntervals_v2([[1, 2], [2, 3]]), 0)

    def test_pathSum(self):
        # 437.Path Sum III
        self.assertEqual(self.sl.pathSum(create_binary_tree([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1]), 8), 3)
        self.assertEqual(self.sl.pathSum(create_binary_tree([1, -2, -3, 1, 3, -2, None, -1]), -1), 4)

    def test_findAnagrams(self):
        # 438.Find All Anagrams in a String
        self.assertListEqual(self.sl.findAnagrams("cbaebabacd", "abc"), [0, 6])
        self.assertListEqual(self.sl.findAnagrams("abab", "ab"), [0, 1, 2])
        self.assertListEqual(self.sl.findAnagrams("aab", "ab"), [1])
        self.assertListEqual(self.sl.findAnagrams("abab", "baddfd"), [])

    def test_arrangeCoins(self):
        # 441.Arranging Coins
        self.assertEqual(self.sl.arrangeCoins(5), 2)
        self.assertEqual(self.sl.arrangeCoins(8), 3)
        self.assertEqual(self.sl.arrangeCoins(1), 1)

    def test_compress(self):
        # 443.String Compression
        self.assertEqual(self.sl.compress(["a", "a", "b", "b", "c", "c", "c"]), 6)
        self.assertEqual(self.sl.compress(["a"]), 1)
        self.assertEqual(self.sl.compress(["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]), 4)

    def test_addTwoNumbers(self):
        # 445.Add Two Numbers II
        l1 = create_linked_list([7, 2, 4, 3])
        l2 = create_linked_list([5, 6, 4])
        ans = create_linked_list([7, 8, 0, 7])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers(l1, l2), ans))
        l1 = create_linked_list([7, 2, 4, 3])
        l2 = create_linked_list([5, 6, 4])
        ans = create_linked_list([7, 8, 0, 7])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers(l2, l1), ans))
        l1 = create_linked_list([2, 4, 3])
        l2 = create_linked_list([5, 6, 4])
        ans = create_linked_list([8, 0, 7])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers(l1, l2), ans))
        l1 = create_linked_list([0])
        l2 = create_linked_list([0])
        ans = create_linked_list([0])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers(l1, l2), ans))
        l1 = create_linked_list([7, 2, 4, 3])
        l2 = create_linked_list([5, 6, 4])
        ans = create_linked_list([7, 8, 0, 7])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers_v2(l1, l2), ans))
        l1 = create_linked_list([7, 2, 4, 3])
        l2 = create_linked_list([5, 6, 4])
        ans = create_linked_list([7, 8, 0, 7])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers_v2(l2, l1), ans))
        l1 = create_linked_list([2, 4, 3])
        l2 = create_linked_list([5, 6, 4])
        ans = create_linked_list([8, 0, 7])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers_v2(l1, l2), ans))
        l1 = create_linked_list([0])
        l2 = create_linked_list([0])
        ans = create_linked_list([0])
        self.assertTrue(eq_linked_list(self.sl.addTwoNumbers_v2(l1, l2), ans))

    def test_deleteNode(self):
        # 450.Delete Node in a BST
        root = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        ans = create_binary_tree([5, 4, 6, 2, None, None, 7])
        self.assertTrue(eq_binary_tree(self.sl.deleteNode(root, 3), ans))
        root = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        ans = create_binary_tree([5, 3, 6, None, 4, None, 7])
        self.assertTrue(eq_binary_tree(self.sl.deleteNode(root, 2), ans))
        root = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        ans = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        self.assertTrue(eq_binary_tree(self.sl.deleteNode(root, 0), ans))
        self.assertIsNone(self.sl.deleteNode(None, 0))
        root = create_binary_tree([2, 1])
        ans = create_binary_tree([1])
        self.assertTrue(eq_binary_tree(self.sl.deleteNode(root, 2), ans))
        root = create_binary_tree([50, 30, 70, None, 40, 60, 80])
        ans = create_binary_tree([60, 30, 70, None, 40, None, 80])
        self.assertTrue(eq_binary_tree(self.sl.deleteNode(root, 50), ans))
        root = create_binary_tree([5, 3, 6, 2, 4, None, 7])
        ans = create_binary_tree([5, 3, 6, 2, 4])
        self.assertTrue(eq_binary_tree(self.sl.deleteNode(root, 7), ans))

    def test_frequencySort(self):
        # 451.Sort Characters By Frequency
        self.assertEqual(self.sl.frequencySort("tree"), "eetr")
        self.assertEqual(self.sl.frequencySort("cccaaa"), "cccaaa")
        self.assertEqual(self.sl.frequencySort("Aabb"), "bbAa")

    def test_findMinArrowShots(self):
        # 452.Minimum Number of Arrows to Burst Balloons
        self.assertEqual(self.sl.findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]), 2)
        self.assertEqual(self.sl.findMinArrowShots([[1, 2], [3, 4], [5, 6], [7, 8]]), 4)
        self.assertEqual(self.sl.findMinArrowShots([[1, 2], [2, 3], [3, 4], [4, 5]]), 2)
        self.assertEqual(self.sl.findMinArrowShots([]), 0)

    def test_repeatedSubstringPattern(self):
        # 459.Repeated Substring Pattern
        self.assertTrue(self.sl.repeatedSubstringPattern("abab"))
        self.assertFalse(self.sl.repeatedSubstringPattern("aba"))
        self.assertFalse(self.sl.repeatedSubstringPattern("abababb"))
        self.assertTrue(self.sl.repeatedSubstringPattern("abcabcabcabc"))

    def test_PredictTheWinner(self):
        # 486.Predict the Winner
        self.assertFalse(self.sl.PredictTheWinner([1, 5, 2]))
        self.assertTrue(self.sl.PredictTheWinner([1, 5, 233, 7]))

    def test_nextGreaterElement(self):
        # 496.Next Greater Element1
        self.assertListEqual(self.sl.nextGreaterElement([4, 1, 2], [1, 3, 4, 2]), [-1, 3, -1])
        self.assertListEqual(self.sl.nextGreaterElement([2, 4], [1, 2, 3, 4]), [3, -1])

    def test_nextGreaterElements(self):
        # 503.Next Greater Element II
        self.assertEqual(self.sl.nextGreaterElements([1, 2, 1]), [2, -1, 2])
        self.assertEqual(self.sl.nextGreaterElements([1, 2, 3, 4, 3]), [2, 3, 4, -1, 4])

    def test_fib(self):
        # 509.Fibonacci Number
        self.assertEqual(self.sl.fib(0), 0)
        self.assertEqual(self.sl.fib(1), 1)
        self.assertEqual(self.sl.fib(2), 1)
        self.assertEqual(self.sl.fib(3), 2)
        self.assertEqual(self.sl.fib(4), 3)

    def test_longestPalindromeSubseq(self):
        # 516.Longest Palindromic Subsequence
        self.assertEqual(self.sl.longestPalindromeSubseq("bbbab"), 4)
        self.assertEqual(self.sl.longestPalindromeSubseq("cbbd"), 2)

    def test_change(self):
        # 518.Coin Change II
        self.assertEqual(self.sl.change(5, [1, 2, 5]), 4)
        self.assertEqual(self.sl.change(3, [2]), 0)
        self.assertEqual(self.sl.change(10, [10]), 1)
        self.assertEqual(self.sl.change_v2(5, [1, 2, 5]), 4)
        self.assertEqual(self.sl.change_v2(3, [2]), 0)
        self.assertEqual(self.sl.change_v2(10, [10]), 1)

    def test_updateMatrix(self):
        # 542.0 1 Matrix
        mat = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        expected = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertEqual(self.sl.updateMatrix(mat), expected)
        mat = [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
        expected = [[0, 0, 0], [0, 1, 0], [1, 2, 1]]
        self.assertEqual(self.sl.updateMatrix(mat), expected)

    def test_diameterOfBinaryTree(self):
        # 543.Diameter of Binary Tree
        root = create_binary_tree([1, 2, 3, 4, 5])
        self.assertEqual(self.sl.diameterOfBinaryTree(root), 3)
        root = create_binary_tree([1, 2])
        self.assertEqual(self.sl.diameterOfBinaryTree(root), 1)

    def test_findCircleNum(self):
        # 547.Number of Provinces
        self.assertEqual(self.sl.findCircleNum([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), 2)
        self.assertEqual(self.sl.findCircleNum([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3)

    def test_nextGreaterElement_v2(self):
        # 556.Next Greater Element III
        self.assertEqual(self.sl.nextGreaterElement_v2(12), 21)
        self.assertEqual(self.sl.nextGreaterElement_v2(21), -1)
        self.assertEqual(self.sl.nextGreaterElement_v2(1234), 1243)
        self.assertEqual(self.sl.nextGreaterElement_v2(230241), 230412)
        self.assertEqual(self.sl.nextGreaterElement_v2(11), -1)
        self.assertEqual(self.sl.nextGreaterElement_v2(2147483486), -1)

    def test_reverseWords(self):
        # 557.Reverse Words in a String 3
        s1 = "Let's take LeetCode contest"
        s2 = "God Ding"
        expected1 = "s'teL ekat edoCteeL tsetnoc"
        expected2 = "doG gniD"
        self.assertEqual(self.sl.reverseWords(s1), expected1)
        self.assertEqual(self.sl.reverseWords(s2), expected2)

    def test_subarraySum(self):
        # 560.Subarray Sum Equals K
        self.assertEqual(self.sl.subarraySum([1, 1, 1], 2), 2)
        self.assertEqual(self.sl.subarraySum([1, 2, 3], 3), 2)
        self.assertEqual(self.sl.subarraySum([1, -1, 3, 4, 5, 6, 7, 8, 9], 9), 2)

    def test_matrixReshape(self):
        # 566.Reshape the Matrix
        mat, r, c = [[1, 2], [3, 4]], 1, 4
        self.assertListEqual(self.sl.matrixReshape(mat, r, c), [[1, 2, 3, 4]])
        r = 2
        self.assertListEqual(self.sl.matrixReshape(mat, r, c), [[1, 2], [3, 4]])

    def test_checkInclusion(self):
        # 567.Permutation in String
        self.assertTrue(self.sl.checkInclusion("ab", "eidbaooo"))
        self.assertTrue(self.sl.checkInclusion("ab", "abcdefg"))
        self.assertFalse(self.sl.checkInclusion("ab", "eidboaoo"))
        self.assertFalse(self.sl.checkInclusion("ab", "a"))

    def test_isSubtree(self):
        # 572.Subtree of Another Tree
        self.assertTrue(self.sl.isSubtree(create_binary_tree([3, 4, 5, 1, 2]), create_binary_tree([4, 1, 2])))
        self.assertFalse(self.sl.isSubtree(create_binary_tree([3, 4, 5, 1, 2]), create_binary_tree([4, 1, 2, 3, 4, 5])))
        self.assertFalse(
            self.sl.isSubtree(
                create_binary_tree([3, 4, 5, 1, 2, None, None, None, None, 0]), create_binary_tree([4, 1, 2])
            )
        )

    def test_minDistance(self):
        # 583.Delete Operation for Two Strings
        self.assertEqual(self.sl.minDistance("sea", "eat"), 2)
        self.assertEqual(self.sl.minDistance("leetcode", "etco"), 4)

    def test_preorder(self):
        # 589.N-ary Tree Preorder Traversal
        pass
