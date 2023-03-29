import unittest

from pysolutions import Pro1001To1200


class TestP1001To1200(unittest.TestCase):
    @property
    def sl(self):
        return Pro1001To1200()

    def test_commonChars(self):
        # 1002.Find Common Characters
        words1 = ["bella", "label", "roller"]
        words2 = ["cool", "lock", "cook"]
        self.assertListEqual(self.sl.commonChars(words1), ["e", "l", "l"])
        self.assertListEqual(self.sl.commonChars(words2), ["c", "o"])
        self.assertListEqual(self.sl.commonChars(["words"]), ["w", "o", "r", "d", "s"])

    def test_lastStoneWeight(self):
        # 1046.Last Stone Weight
        self.assertEqual(self.sl.lastStoneWeight([2, 7, 4, 1, 8, 1]), 1)
        self.assertEqual(self.sl.lastStoneWeight([1]), 1)
        self.assertEqual(self.sl.lastStoneWeight([1, 1, 2, 2]), 0)
