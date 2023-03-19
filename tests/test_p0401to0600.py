import unittest

from pysolutions import Pro0401To0600


class TestP0401To0600(unittest.TestCase):
    @property
    def sl(self):
        return Pro0401To0600()

    def test_longestPalindrom(self):
        # 409.Longest Palindrome
        self.assertEqual(self.sl.longestPalindrome("abccccdd"), 7)
        self.assertEqual(self.sl.longestPalindrome("a"), 1)

    def test_nextGreaterElement(self):
        # 496.Next Greater Element1
        self.assertListEqual(self.sl.nextGreaterElement([4, 1, 2], [1, 3, 4, 2]), [-1, 3, -1])
        self.assertListEqual(self.sl.nextGreaterElement([2, 4], [1, 2, 3, 4]), [3, -1])

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
        self.assertFalse(self.sl.checkInclusion("ab", "eidboaoo"))
