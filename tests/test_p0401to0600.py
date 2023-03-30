import unittest

from pysolutions import Pro0401To0600
from pysolutions.utils import create_binary_tree


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

    def test_characterReplacement(self):
        # 424.Longest Repeating Character Replacement
        self.assertEqual(self.sl.characterReplacement("ABAB", 2), 4)
        self.assertEqual(self.sl.characterReplacement("AABABBA", 1), 4)
        self.assertEqual(self.sl.characterReplacement("AABABBA", 2), 5)

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

    def test_repeatedSubstringPattern(self):
        # 459.Repeated Substring Pattern
        self.assertTrue(self.sl.repeatedSubstringPattern("abab"))
        self.assertFalse(self.sl.repeatedSubstringPattern("aba"))
        self.assertFalse(self.sl.repeatedSubstringPattern("abababb"))
        self.assertTrue(self.sl.repeatedSubstringPattern("abcabcabcabc"))

    def test_nextGreaterElement(self):
        # 496.Next Greater Element1
        self.assertListEqual(self.sl.nextGreaterElement([4, 1, 2], [1, 3, 4, 2]), [-1, 3, -1])
        self.assertListEqual(self.sl.nextGreaterElement([2, 4], [1, 2, 3, 4]), [3, -1])

    def test_fib(self):
        # 509.Fibonacci Number
        self.assertEqual(self.sl.fib(0), 0)
        self.assertEqual(self.sl.fib(1), 1)
        self.assertEqual(self.sl.fib(2), 1)
        self.assertEqual(self.sl.fib(3), 2)
        self.assertEqual(self.sl.fib(4), 3)

    def test_updateMatrix(self):
        # 542.0 1 Matrix
        mat = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        expected = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertEqual(self.sl.updateMatrix(mat), expected)
        mat = [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
        expected = [[0, 0, 0], [0, 1, 0], [1, 2, 1]]
        self.assertEqual(self.sl.updateMatrix(mat), expected)

    def test_reverseWords(self):
        # 557.Reverse Words in a String 3
        s1 = "Let's take LeetCode contest"
        s2 = "God Ding"
        expected1 = "s'teL ekat edoCteeL tsetnoc"
        expected2 = "doG gniD"
        self.assertEqual(self.sl.reverseWords(s1), expected1)
        self.assertEqual(self.sl.reverseWords(s2), expected2)

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

    def test_preorder(self):
        # 589.N-ary Tree Preorder Traversal
        pass
