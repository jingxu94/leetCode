import unittest

from pysolutions import Pro2401To2600


class TestP2401To2600(unittest.TestCase):
    @property
    def sl(self):
        return Pro2401To2600()

    def test_splitNum(self):
        # 2578.Split With Minimum Sum
        self.assertEqual(self.sl.splitNum(4325), 59)
        self.assertEqual(self.sl.splitNum(687), 75)
