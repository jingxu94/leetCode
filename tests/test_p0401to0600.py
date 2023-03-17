import unittest

from pysolutions import Pro0401To0600


class TestP0401To0600(unittest.TestCase):
    @property
    def sl(self):
        return Pro0401To0600()

    def test_reverseWords(self):
        # 557.Reverse Words in a String 3
        s1 = "Let's take LeetCode contest"
        s2 = "God Ding"
        expected1 = "s'teL ekat edoCteeL tsetnoc"
        expected2 = "doG gniD"
        self.assertEqual(self.sl.reverseWords(s1), expected1)
        self.assertEqual(self.sl.reverseWords(s2), expected2)
