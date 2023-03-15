import unittest

from pysolutions import Pro1401To1600


class TestP1401To1600(unittest.TestCase):
    @property
    def sl(self):
        return Pro1401To1600()

    def test_runningSum(self):
        # 1480.Running Sum of 1d Array
        self.assertEqual(self.sl.runningSum([1, 2, 3, 4]), [1, 3, 6, 10])
        self.assertEqual(self.sl.runningSum([1, 1, 1, 1, 1]), [1, 2, 3, 4, 5])
        self.assertEqual(self.sl.runningSum([3, 1, 2, 10, 1]), [3, 4, 6, 16, 17])
