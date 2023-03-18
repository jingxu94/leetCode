import unittest

from pysolutions import Pro0601To0800


class TestP0601To0800(unittest.TestCase):
    @property
    def sl(self):
        return Pro0601To0800()

    def test_search(self):
        # 704.Binary Search
        self.assertEqual(self.sl.search([-1, 0, 3, 5, 9, 12], 9), 4)
        self.assertEqual(self.sl.search([-1, 0, 3, 5, 9, 12], 2), -1)

    def test_pivotIndex(self):
        # 724.Find Pivot Index
        self.assertEqual(self.sl.pivotIndex([1, 7, 3, 6, 5, 6]), 3)
        self.assertEqual(self.sl.pivotIndex([1, 2, 3]), -1)
        self.assertEqual(self.sl.pivotIndex([2, 1, -1]), 0)

    def test_nextGreatestLetter(self):
        # 744.Find Smallest Letter Greater Than Target
        self.assertEqual(self.sl.nextGreatestLetter(["c", "f", "j"], "a"), "c")
        self.assertEqual(self.sl.nextGreatestLetter(["c", "f", "j"], "c"), "f")
        self.assertEqual(self.sl.nextGreatestLetter(["x", "x", "y", "y"], "z"), "x")
