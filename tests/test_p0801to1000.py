import unittest

from pysolutions import Pro0801To1000


class TestP0801To1000(unittest.TestCase):
    @property
    def sl(self):
        return Pro0801To1000()

    def test_minEatingSpeed(self):
        # 875.Koko Eating Bananas
        piles1 = [3, 6, 7, 11]
        piles2 = [30, 11, 23, 4, 20]
        piles3 = [30, 11, 23, 4, 20]
        h1, h2, h3 = 8, 5, 6
        expres1, expres2, expres3 = 4, 30, 23
        self.assertEqual(self.sl.minEatingSpeed(piles1, h1), expres1)
        self.assertEqual(self.sl.minEatingSpeed(piles2, h2), expres2)
        self.assertEqual(self.sl.minEatingSpeed(piles3, h3), expres3)

    def test_sortArray(self):
        # 912.Sort an Array
        nums1 = [5, 2, 3, 1]
        nums2 = [5, 1, 1, 2, 0, 0]
        expres1 = [1, 2, 3, 5]
        expres2 = [0, 0, 1, 1, 2, 5]
        self.assertEqual(self.sl.sortArray(nums1), expres1)
        self.assertEqual(self.sl.sortArray(nums2), expres2)
