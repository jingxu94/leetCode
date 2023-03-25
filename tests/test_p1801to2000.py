import unittest

from pysolutions import Pro1801To2000


class TestP1801To2000(unittest.TestCase):
    @property
    def sl(self):
        return Pro1801To2000()

    def test_arraySign(self):
        # 1822.Sign of the Product of an Array
        self.assertEqual(self.sl.arraySign([-1, -2, -3, -4, 3, 2, 1]), 1)
        self.assertEqual(self.sl.arraySign([1, 5, 0, 2, -3]), 0)
        self.assertEqual(self.sl.arraySign([-1, 1, -1, 1, -1]), -1)

    def test_maxDistance(self):
        # 1855.Maximum Distance Between a Pair of Values
        self.assertEqual(self.sl.maxDistance([55, 30, 5, 4, 2], [100, 20, 10, 10, 5]), 2)
        self.assertEqual(self.sl.maxDistance([2, 2, 2], [10, 10, 1]), 1)
        self.assertEqual(self.sl.maxDistance([30, 29, 19, 5], [25, 25, 25, 25, 25]), 2)
