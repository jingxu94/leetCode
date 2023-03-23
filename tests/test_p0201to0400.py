import unittest

from pysolutions import Pro0201To0400
from pysolutions.utils import create_binary_tree, create_linked_list, eq_binary_tree, eq_linked_list


class TestP0201To0400(unittest.TestCase):
    @property
    def sl(self):
        return Pro0201To0400()

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

    def test_containDuplicate(self):
        # 217.Contains Duplicate
        self.assertTrue(self.sl.containsDuplicate([1, 2, 3, 1]))
        self.assertTrue(self.sl.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))
        self.assertFalse(self.sl.containsDuplicate([1, 2, 3, 4]))

    def test_isPalindrome(self):
        # 234.Palindrome Linked List
        self.assertTrue(self.sl.isPalindrome(create_linked_list([1, 2, 2, 1])))
        self.assertFalse(self.sl.isPalindrome(create_linked_list([1, 2])))

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

    def test_maxSlidingWindow(self):
        # 239.Sliding Window Maximum
        nums = [1, 3, -1, -3, 5, 3, 6, 7]
        k = 3
        ans = self.sl.maxSlidingWindow(nums, k)
        expected = [3, 3, 5, 5, 6, 7]
        self.assertListEqual(ans, expected)
        self.assertListEqual(self.sl.maxSlidingWindow([], k), [])

    def test_isAnagram(self):
        # 242.Valid Anagram
        self.assertTrue(self.sl.isAnagram("anagram", "nagaram"))
        self.assertFalse(self.sl.isAnagram("rat", "car"))

    def test_firstBadVersion(self):
        # 278.First Bad Version
        # There's no inner function isBadVersion()
        pass

    def test_moveZeroes(self):
        # 283.Move Zeroes
        self.assertListEqual(self.sl.moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
        self.assertEqual(self.sl.moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
        self.assertListEqual(self.sl.moveZeroes([0]), [0])
        self.assertEqual(self.sl.moveZeroes([0]), [0])

    def test_reverseString(self):
        # 344.Reverse String
        self.assertEqual(self.sl.reverseString(["a", "b"]), ["b", "a"])

    def test_intersect(self):
        # 350.Intersection of Two Arrays 2
        self.assertEqual(self.sl.intersect([1, 2, 2, 1], [2, 2]), [2, 2])
        self.assertEqual(self.sl.intersect([4, 9, 5], [9, 4, 9, 8, 4]), [4, 9])

    def test_isPerfectSquare(self):
        # 367.Valid Perfect Square
        self.assertTrue(self.sl.isPerfectSquare(16))
        self.assertFalse(self.sl.isPerfectSquare(14))

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
