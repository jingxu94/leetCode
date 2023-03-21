import unittest

from pysolutions import Pro2201To2400


class TestP2201To2400(unittest.TestCase):
    @property
    def sl(self):
        return Pro2201To2400()

    def test_zeroFilledSubarray(self):
        # 2348.Number of Zero-Filled Subarrays
        self.assertEqual(self.sl.zeroFilledSubarray([1, 3, 0, 0, 2, 0, 0, 4]), 6)
        self.assertEqual(self.sl.zeroFilledSubarray([0, 0, 0, 2, 0, 0]), 9)
        self.assertEqual(self.sl.zeroFilledSubarray([2, 10, 2019]), 0)
