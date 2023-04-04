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

    def test_maxScoreSightseeingPair(self):
        # 1014.Best Sightseeing Pair
        self.assertEqual(self.sl.maxScoreSightseeingPair([8, 1, 5, 2, 6]), 11)
        self.assertEqual(self.sl.maxScoreSightseeingPair([1, 2]), 2)

    def test_lastStoneWeight(self):
        # 1046.Last Stone Weight
        self.assertEqual(self.sl.lastStoneWeight([2, 7, 4, 1, 8, 1]), 1)
        self.assertEqual(self.sl.lastStoneWeight([1]), 1)
        self.assertEqual(self.sl.lastStoneWeight([1, 1, 2, 2]), 0)

    def test_tribonacci(self):
        # 1137.N-th Tribonacci Number
        self.assertEqual(self.sl.tribonacci(1), 1)
        self.assertEqual(self.sl.tribonacci(2), 1)
        self.assertEqual(self.sl.tribonacci(4), 4)
        self.assertEqual(self.sl.tribonacci(25), 1389537)
