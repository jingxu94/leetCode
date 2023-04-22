import unittest

from pysolutions import Pro2001To2200


class TestP2001To2200(unittest.TestCase):
    @property
    def sl(self):
        return Pro2001To2200()

    def test_longestPalindrome(self):
        # 2131.Longest Palindrome by Concatenating Two Letter Words
        self.assertEqual(self.sl.longestPalindrome(["lc", "cl", "gg"]), 6)
        self.assertEqual(self.sl.longestPalindrome(["ab","ty","yt","lc","cl","ab"]), 8)
        self.assertEqual(self.sl.longestPalindrome(["cc","ll","xx"]), 2)

    def test_minimumTime(self):
        # 2187.Minimum Time to Complete Trips
        self.assertEqual(self.sl.minimumTime([1, 2, 3], 5), 3)
        self.assertEqual(self.sl.minimumTime([2], 1), 2)
