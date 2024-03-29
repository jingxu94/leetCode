import unittest

from pysolutions import Pro0201To0400
from pysolutions.utils import create_binary_tree, create_linked_list, eq_binary_tree, eq_linked_list


class TestP0201To0400(unittest.TestCase):
    @property
    def sl(self):
        return Pro0201To0400()

    def test_rangeBitwiseAnd(self):
        # 201.Bitwise AND of Numbers Range
        self.assertEqual(self.sl.rangeBitwiseAnd(5, 7), 4)
        self.assertEqual(self.sl.rangeBitwiseAnd(0, 0), 0)
        self.assertEqual(self.sl.rangeBitwiseAnd(1, 2147483647), 0)

    def test_isHappy(self):
        # 202.Happy Number
        self.assertTrue(self.sl.isHappy(19))
        self.assertFalse(self.sl.isHappy(2))

    def test_removeElements(self):
        # 203.Remove Linked List Elements
        head = create_linked_list([1, 2, 6, 3, 4, 5, 6])
        ans = self.sl.removeElements(head, 6)
        self.assertTrue(eq_linked_list(create_linked_list([1, 2, 3, 4, 5]), ans))
        head = create_linked_list([])
        ans = self.sl.removeElements(head, 1)
        self.assertTrue(eq_linked_list(create_linked_list([]), ans))
        head = create_linked_list([7, 7, 7, 7])
        ans = self.sl.removeElements(head, 7)
        self.assertTrue(eq_linked_list(create_linked_list([]), ans))

    def test_isIsomorphic(self):
        # 205.Isomorphic Strings
        self.assertTrue(self.sl.isIsomorphic("egg", "add"))
        self.assertFalse(self.sl.isIsomorphic("foo", "bar"))
        self.assertFalse(self.sl.isIsomorphic("foo", "bbar"))
        self.assertTrue(self.sl.isIsomorphic("paper", "title"))
        self.assertFalse(self.sl.isIsomorphic("badc", "baba"))

    def test_reverseList(self):
        # 206.Reverse Linked List
        self.assertTrue(
            eq_linked_list(
                self.sl.reverseList(create_linked_list([1, 2, 3, 4, 5])), create_linked_list([5, 4, 3, 2, 1])
            )
        )
        self.assertTrue(eq_linked_list(self.sl.reverseList(create_linked_list([1, 2])), create_linked_list([2, 1])))
        self.assertTrue(eq_linked_list(self.sl.reverseList(create_linked_list([])), create_linked_list([])))

    def test_canFinish(self):
        # 207.Course Schedule
        self.assertTrue(self.sl.canFinish(2, [[1, 0]]))
        self.assertFalse(self.sl.canFinish(2, [[1, 0], [0, 1]]))
        self.assertTrue(self.sl.canFinish(5, [[1, 4], [2, 4], [3, 1], [3, 2]]))
        self.assertTrue(self.sl.canFinish_v2(2, [[1, 0]]))
        self.assertFalse(self.sl.canFinish_v2(2, [[1, 0], [0, 1]]))
        self.assertTrue(self.sl.canFinish_v2(5, [[1, 4], [2, 4], [3, 1], [3, 2]]))

    def test_minSubArrayLen(self):
        # 209.Minimum Size Subarray Sum
        self.assertEqual(self.sl.minSubArrayLen(7, [2, 3, 1, 2, 4, 3]), 2)
        self.assertEqual(self.sl.minSubArrayLen(4, [1, 4, 4]), 1)
        self.assertEqual(self.sl.minSubArrayLen(11, [1, 1, 1, 1, 1, 1, 1, 1]), 0)

    def test_findOrder(self):
        # 210.Course Schedule II
        self.assertEqual(self.sl.findOrder(2, [[1, 0]]), [0, 1])
        self.assertEqual(self.sl.findOrder(4, [[1, 0], [2, 0], [3, 1], [3, 2]]), [0, 1, 2, 3])
        self.assertEqual(self.sl.findOrder(1, []), [0])
        self.assertEqual(self.sl.findOrder(3, [[1, 0], [1, 2], [0, 1]]), [])

    def test_findWords(self):
        # 212.Word Search II
        self.assertEqual(
            sorted(
                self.sl.findWords(
                    [["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]],
                    ["oath", "pea", "eat", "rain"],
                )
            ),
            sorted(["eat", "oath"]),
        )
        self.assertEqual(sorted(self.sl.findWords([["a", "b"], ["c", "d"]], ["abcb"])), sorted([]))

    def test_rob(self):
        # 213.House Robber II
        self.assertEqual(self.sl.rob([2, 3, 2]), 3)
        self.assertEqual(self.sl.rob([1, 2, 3, 1]), 4)
        self.assertEqual(self.sl.rob([1, 2, 3]), 3)
        self.assertEqual(self.sl.rob([3]), 3)
        self.assertEqual(self.sl.rob([]), 0)

    def test_findKthLargest(self):
        # 215.Kth Largest Element in an Array
        self.assertEqual(self.sl.findKthLargest([3, 2, 1, 5, 6, 4], 2), 5)
        self.assertEqual(self.sl.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4), 4)
        self.assertEqual(self.sl.findKthLargest_v2([3, 2, 1, 5, 6, 4], 2), 5)
        self.assertEqual(self.sl.findKthLargest_v2([3, 2, 3, 1, 2, 4, 5, 5, 6], 4), 4)

    def test_combinationSum3(self):
        # 216.Combination Sum III
        self.assertEqual(self.sl.combinationSum3(3, 7), [[1, 2, 4]])
        self.assertEqual(self.sl.combinationSum3(3, 9), [[1, 2, 6], [1, 3, 5], [2, 3, 4]])
        self.assertEqual(self.sl.combinationSum3(4, 1), [])

    def test_containDuplicate(self):
        # 217.Contains Duplicate
        self.assertTrue(self.sl.containsDuplicate([1, 2, 3, 1]))
        self.assertTrue(self.sl.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))
        self.assertFalse(self.sl.containsDuplicate([1, 2, 3, 4]))

    def test_maximalSquare(self):
        # 221.Maximal Square
        self.assertEqual(
            self.sl.maximalSquare(
                [
                    ["1", "0", "1", "0", "0"],
                    ["1", "0", "1", "1", "1"],
                    ["1", "1", "1", "1", "1"],
                    ["1", "0", "0", "1", "0"],
                ]
            ),
            4,
        )
        self.assertEqual(self.sl.maximalSquare([["0", "1"], ["1", "0"]]), 1)
        self.assertEqual(self.sl.maximalSquare([["0"]]), 0)

    def test_invertTree(self):
        # 226.Invert Binary Tree
        root = create_binary_tree([4, 2, 7, 1, 3, 6, 9])
        expected = create_binary_tree([4, 7, 2, 9, 6, 3, 1])
        self.assertTrue(eq_binary_tree(self.sl.invertTree(root), expected))
        root = create_binary_tree([2, 1, 3])
        expected = create_binary_tree([2, 3, 1])
        self.assertTrue(eq_binary_tree(self.sl.invertTree(root), expected))
        root = create_binary_tree([1])
        expected = create_binary_tree([1])
        self.assertTrue(eq_binary_tree(self.sl.invertTree(root), expected))
        root = create_binary_tree([])
        expected = create_binary_tree([])
        self.assertTrue(eq_binary_tree(self.sl.invertTree(root), expected))

    def test_summaryRanges(self):
        # 228.Summary Ranges
        self.assertEqual(self.sl.summaryRanges([0, 1, 2, 4, 5, 7]), ["0->2", "4->5", "7"])
        self.assertEqual(self.sl.summaryRanges([0, 2, 3, 4, 6, 8, 9]), ["0", "2->4", "6", "8->9"])

    def test_kthSmallest(self):
        # 230.Kth Smallest Element in a BST
        root = create_binary_tree([3, 1, 4, None, 2])
        self.assertEqual(self.sl.kthSmallest(root, 1), 1)
        self.assertEqual(self.sl.kthSmallest(root, 2), 2)
        self.assertEqual(self.sl.kthSmallest(root, 3), 3)
        self.assertEqual(self.sl.kthSmallest(root, 4), 4)
        root = create_binary_tree([5, 3, 6, 2, 4, None, None, 1])
        self.assertEqual(self.sl.kthSmallest(root, 1), 1)
        self.assertEqual(self.sl.kthSmallest(root, 2), 2)
        self.assertEqual(self.sl.kthSmallest(root, 3), 3)
        self.assertEqual(self.sl.kthSmallest(root, 4), 4)
        self.assertEqual(self.sl.kthSmallest(root, 5), 5)

    def test_isPowerOfTwo(self):
        # 231.Power of Two
        self.assertFalse(self.sl.isPowerOfTwo(0))
        self.assertTrue(self.sl.isPowerOfTwo(1))
        self.assertTrue(self.sl.isPowerOfTwo(16))
        self.assertFalse(self.sl.isPowerOfTwo(218))

    def test_isPalindrome(self):
        # 234.Palindrome Linked List
        self.assertTrue(self.sl.isPalindrome(create_linked_list([1, 2, 2, 1])))
        self.assertFalse(self.sl.isPalindrome(create_linked_list([1, 2])))
        self.assertFalse(self.sl.isPalindrome(create_linked_list([])))

    def test_lowestCommonAncestor(self):
        # 235.Lowest Common Ancestor of a Binary Search Tree
        root = create_binary_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
        p, q = root.left, root.right
        expected = root
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, p, q), expected))
        root = create_binary_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
        p, q = root.left, root.left.right
        expected = root.left
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, p, q), expected))
        root = create_binary_tree([2, 1])
        p, q = root, root.left
        expected = root
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, p, q), expected))
        root = create_binary_tree([2, None, 1])
        p, q = root, root.right
        expected = root
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, p, q), expected))
        self.assertEqual(self.sl.lowestCommonAncestor(None, p, q), None)
        root = create_binary_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
        p, q = root.left.right.left, root.left.right.right
        expected = root.left.right
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, p, q), expected))
        root = create_binary_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
        p, q = root.left.right.left, None
        expected = p
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, p, q), expected))
        self.assertTrue(eq_binary_tree(self.sl.lowestCommonAncestor(root, q, p), expected))

    def test_productExceptSelf(self):
        # 238.Product of Array Except Self
        self.assertEqual(self.sl.productExceptSelf([1, 2, 3, 4]), [24, 12, 8, 6])
        self.assertEqual(self.sl.productExceptSelf([-1, 1, 0, -3, 3]), [0, 0, 9, 0, 0])
        self.assertEqual(self.sl.productExceptSelf([1, 2, 0, 0]), [0, 0, 0, 0])

    def test_maxSlidingWindow(self):
        # 239.Sliding Window Maximum
        nums = [1, 3, -1, -3, 5, 3, 6, 7]
        k = 3
        ans = self.sl.maxSlidingWindow(nums, k)
        expected = [3, 3, 5, 5, 6, 7]
        self.assertListEqual(ans, expected)
        self.assertListEqual(self.sl.maxSlidingWindow([], k), [])

    def test_searchMatrix(self):
        # 240.Search a 2D Matrix II
        matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
        self.assertTrue(self.sl.searchMatrix(matrix, 5))
        self.assertFalse(self.sl.searchMatrix(matrix, 20))

    def test_isAnagram(self):
        # 242.Valid Anagram
        self.assertTrue(self.sl.isAnagram("anagram", "nagaram"))
        self.assertFalse(self.sl.isAnagram("rat", "car"))

    def test_addDigits(self):
        # 258.Add Digits
        self.assertEqual(self.sl.addDigits(38), 2)
        self.assertEqual(self.sl.addDigits(0), 0)

    def test_nthUglyNumber(self):
        # 264.Ugly Number II
        self.assertEqual(self.sl.nthUglyNumber(10), 12)
        self.assertEqual(self.sl.nthUglyNumber(1), 1)

    def test_firstBadVersion(self):
        # 278.First Bad Version
        # There's no inner function isBadVersion()
        pass

    def test_numSquares(self):
        # 279.Perfect Squares
        self.assertEqual(self.sl.numSquares(12), 3)
        self.assertEqual(self.sl.numSquares(13), 2)

    def test_moveZeroes(self):
        # 283.Move Zeroes
        self.assertListEqual(self.sl.moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
        self.assertEqual(self.sl.moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
        self.assertListEqual(self.sl.moveZeroes([0]), [0])
        self.assertEqual(self.sl.moveZeroes([0]), [0])

    def test_findDuplicate(self):
        # 287.Find the Duplicate Number
        self.assertEqual(self.sl.findDuplicate([1, 3, 4, 2, 2]), 2)
        self.assertEqual(self.sl.findDuplicate([3, 1, 3, 4, 2]), 3)

    def test_wordPattern(self):
        # 290.Word Pattern
        self.assertTrue(self.sl.wordPattern("abba", "dog cat cat dog"))
        self.assertFalse(self.sl.wordPattern("abba", "dog cat cat fish"))
        self.assertFalse(self.sl.wordPattern("aaaa", "dog cat cat dog"))
        self.assertFalse(self.sl.wordPattern("aba", "dog cat cat"))
        self.assertFalse(self.sl.wordPattern("abab", "dog cat cat"))
        self.assertTrue(self.sl.wordPattern("abc", "b c a"))

    def test_getHint(self):
        # 299.Bulls and Cows
        self.assertEqual(self.sl.getHint("1807", "7810"), "1A3B")
        self.assertEqual(self.sl.getHint("1123", "0111"), "1A1B")

    def test_lengthOfLIS(self):
        # 300.Longest Increasing Subsequence
        self.assertEqual(self.sl.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]), 4)
        self.assertEqual(self.sl.lengthOfLIS([0, 1, 0, 3, 2, 3]), 4)
        self.assertEqual(self.sl.lengthOfLIS([7, 7, 7, 7, 7, 7, 7]), 1)

    def test_maxProfit(self):
        # 309.Best Time to Buy and Sell Stock with Cooldown
        self.assertEqual(self.sl.maxProfit([1, 2, 3, 0, 2]), 3)
        self.assertEqual(self.sl.maxProfit([1]), 0)

    def test_bulbSwitch(self):
        # 319.Bulb Switcher
        self.assertEqual(self.sl.bulbSwitch(3), 1)
        self.assertEqual(self.sl.bulbSwitch(9999999), 3162)

    def test_coinChange(self):
        # 322.Coin Change
        self.assertEqual(self.sl.coinChange([1, 2, 5], 11), 3)
        self.assertEqual(self.sl.coinChange([2], 3), -1)
        self.assertEqual(self.sl.coinChange([1], 0), 0)

    def test_oddEvenList(self):
        # 328.Odd Even Linked List
        head = create_linked_list([1, 2, 3, 4, 5])
        ans = self.sl.oddEvenList(head)
        expected = create_linked_list([1, 3, 5, 2, 4])
        self.assertTrue(eq_linked_list(ans, expected))
        head = create_linked_list([2, 1, 3, 5, 6, 4, 7])
        ans = self.sl.oddEvenList(head)
        expected = create_linked_list([2, 3, 6, 7, 1, 5, 4])
        self.assertTrue(eq_linked_list(ans, expected))
        self.assertEqual(self.sl.oddEvenList(None), None)

    def test_increasingTriplet(self):
        # 334.Increasing Triplet Subsequence
        self.assertTrue(self.sl.increasingTriplet([1, 2, 3, 4, 5]))
        self.assertFalse(self.sl.increasingTriplet([5, 4, 3, 2, 1]))
        self.assertTrue(self.sl.increasingTriplet([2, 1, 5, 0, 4, 6]))
        self.assertTrue(self.sl.increasingTriplet([20, 100, 10, 12, 5, 13]))

    def test_countBits(self):
        # 338.Counting Bits
        self.assertEqual(self.sl.countBits(2), [0, 1, 1])
        self.assertEqual(self.sl.countBits(5), [0, 1, 1, 2, 1, 2])

    def test_integerBreak(self):
        # 343.Integer Break
        self.assertEqual(self.sl.integerBreak(2), 1)
        self.assertEqual(self.sl.integerBreak(3), 2)
        self.assertEqual(self.sl.integerBreak(5), 6)
        self.assertEqual(self.sl.integerBreak(9), 27)
        self.assertEqual(self.sl.integerBreak(10), 36)

    def test_reverseString(self):
        # 344.Reverse String
        self.assertEqual(self.sl.reverseString(["a", "b"]), ["b", "a"])

    def test_reverseVowels(self):
        # 345.Reverse Vowels of a String
        self.assertEqual(self.sl.reverseVowels("hello"), "holle")
        self.assertEqual(self.sl.reverseVowels("leetcode"), "leotcede")

    def test_topKFrequent(self):
        # 347.Top K Frequent Elements
        self.assertEqual(self.sl.topKFrequent([1, 1, 1, 2, 2, 3], 2), [1, 2])
        self.assertEqual(self.sl.topKFrequent([1], 1), [1])

    def test_intersect(self):
        # 350.Intersection of Two Arrays 2
        self.assertEqual(self.sl.intersect([1, 2, 2, 1], [2, 2]), [2, 2])
        self.assertEqual(self.sl.intersect([4, 9, 5], [9, 4, 9, 8, 4]), [4, 9])

    def test_isPerfectSquare(self):
        # 367.Valid Perfect Square
        self.assertTrue(self.sl.isPerfectSquare(16))
        self.assertFalse(self.sl.isPerfectSquare(14))

    def test_kSmallestPairs(self):
        # 373.Find K Pairs with Smallest Sums
        self.assertListEqual(self.sl.kSmallestPairs([1, 7, 11], [2, 4, 6], 3), [[1, 6], [1, 2], [1, 4]])
        self.assertListEqual(self.sl.kSmallestPairs([1, 1, 2], [1, 2, 3], 2), [[1, 1], [1, 1]])
        self.assertListEqual(self.sl.kSmallestPairs([1, 2], [3], 3), [[2, 3], [1, 3]])
        self.assertListEqual(self.sl.kSmallestPairs([1, 2], [], 3), [])

    def test_wiggleMaxLength(self):
        # 376.Wiggle Subsequence
        self.assertEqual(self.sl.wiggleMaxLength([1, 7, 4, 9, 2, 5]), 6)
        self.assertEqual(self.sl.wiggleMaxLength([2]), 1)
        self.assertEqual(self.sl.wiggleMaxLength([0, 0]), 1)
        self.assertEqual(self.sl.wiggleMaxLength([1, 17, 5, 10, 13, 15, 10, 5, 16, 8]), 7)
        self.assertEqual(self.sl.wiggleMaxLength([1, 2, 3, 4, 5, 6, 7, 8, 9]), 2)

    def test_combinationSum4(self):
        # 377.Combination Sum IV
        self.assertEqual(self.sl.combinationSum4([1, 2, 3], 4), 7)
        self.assertEqual(self.sl.combinationSum4([1, 2, 3], 32), 181997601)
        self.assertEqual(self.sl.combinationSum4([9], 3), 0)

    def test_canConstruct(self):
        # 383.Ransom Note
        self.assertFalse(self.sl.canConstruct("a", "b"))
        self.assertFalse(self.sl.canConstruct("aa", "ab"))
        self.assertTrue(self.sl.canConstruct("aa", "aab"))

    def test_firstUniqChar(self):
        # 387.First Unique Character in a String
        self.assertEqual(self.sl.firstUniqChar("leetcode"), 0)
        self.assertEqual(self.sl.firstUniqChar("aabbc"), 4)
        self.assertEqual(self.sl.firstUniqChar("loveleetcode"), 2)
        self.assertEqual(self.sl.firstUniqChar("aabb"), -1)
        self.assertEqual(self.sl.firstUniqChar(""), -1)
        self.assertEqual(self.sl.firstUniqChar("abcdef"), 0)

    def test_findTheDifference(self):
        # 389.Find the Different
        self.assertEqual(self.sl.findTheDifference("abcd", "abcde"), "e")
        self.assertEqual(self.sl.findTheDifference("", "y"), "y")
        self.assertEqual(self.sl.findTheDifference("yyyww", "yyyww"), "")

    def test_isSubsequence(self):
        # 392.Is Subsequence
        self.assertTrue(self.sl.isSubsequence("abc", "ahbgdc"))
        self.assertFalse(self.sl.isSubsequence("axc", "ahbgdc"))

    def test_decodeString(self):
        # 394.Decode String
        self.assertEqual(self.sl.decodeString("3[a]2[bc]"), "aaabcbc")
        self.assertEqual(self.sl.decodeString("3[a2[c]]"), "accaccacc")
        self.assertEqual(self.sl.decodeString("2[abc]3[cd]ef"), "abcabccdcdcdef")

    def test_calcEquation(self):
        # 399.Evaluate Division
        equations = [["a", "b"], ["b", "c"]]
        values = [2.0, 3.0]
        queries = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]
        ans = [6.0, 0.5, -1.0, 1.0, -1.0]
        self.assertEqual(self.sl.calcEquation(equations, values, queries), ans)
        equations = [["a", "b"], ["b", "c"], ["bc", "cd"]]
        values = [1.5, 2.5, 5.0]
        queries = [["a", "c"], ["c", "b"], ["bc", "cd"], ["cd", "bc"]]
        ans = [3.75, 0.4, 5.0, 0.2]
        self.assertEqual(self.sl.calcEquation(equations, values, queries), ans)
        equations = [["a", "b"]]
        values = [0.5]
        queries = [["a", "b"], ["b", "a"], ["a", "c"], ["x", "y"]]
        ans = [0.5, 2.0, -1.0, -1.0]
        self.assertEqual(self.sl.calcEquation(equations, values, queries), ans)
