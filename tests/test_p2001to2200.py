import unittest

from pysolutions import Pro2001To2200


class TestP2001To2200(unittest.TestCase):
    @property
    def sl(self):
        return Pro2001To2200()

    def test_minimumTime(self):
        # 2187.Minimum Time to Complete Trips
        self.assertEqual(self.sl.minimumTime([1, 2, 3], 5), 3)
        self.assertEqual(self.sl.minimumTime([2], 1), 2)
