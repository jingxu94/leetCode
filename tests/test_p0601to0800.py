import unittest

from pysolutions import Pro0601To0800


class TestP0601To0800(unittest.TestCase):
    @property
    def sl(self):
        return Pro0601To0800()

    def test_canPlaceFlowers(self):
        # 605.Can Place Flowers
        self.assertTrue(self.sl.canPlaceFlowers([1, 0, 0, 0, 1], 1))
        self.assertFalse(self.sl.canPlaceFlowers([1, 0, 0, 0, 1], 2))

    def test_maxAreaOfIsland(self):
        # 695.Max Area of Island
        grid = [
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        ]
        self.assertEqual(self.sl.maxAreaOfIsland(grid), 6)
        self.assertEqual(self.sl.maxAreaOfIsland([[0, 0, 0, 0, 0, 0, 0, 0]]), 0)

    def test_search(self):
        # 704.Binary Search
        self.assertEqual(self.sl.search([-1, 0, 3, 5, 9, 12], 9), 4)
        self.assertEqual(self.sl.search([-1, 0, 3, 5, 9, 12], 2), -1)

    def test_pivotIndex(self):
        # 724.Find Pivot Index
        self.assertEqual(self.sl.pivotIndex([1, 7, 3, 6, 5, 6]), 3)
        self.assertEqual(self.sl.pivotIndex([1, 2, 3]), -1)
        self.assertEqual(self.sl.pivotIndex([2, 1, -1]), 0)

    def test_floodFill(self):
        # 733.Flood Fill
        image, sr, sc, color = [[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2
        expected = [[2, 2, 2], [2, 2, 0], [2, 0, 1]]
        self.assertListEqual(self.sl.floodFill(image, sr, sc, color), expected)
        image, sr, sc, color = [[0, 0, 0], [0, 0, 0]], 0, 0, 0
        expected = [[0, 0, 0], [0, 0, 0]]
        self.assertListEqual(self.sl.floodFill(image, sr, sc, color), expected)

    def test_nextGreatestLetter(self):
        # 744.Find Smallest Letter Greater Than Target
        self.assertEqual(self.sl.nextGreatestLetter(["c", "f", "j"], "a"), "c")
        self.assertEqual(self.sl.nextGreatestLetter(["c", "f", "j"], "c"), "f")
        self.assertEqual(self.sl.nextGreatestLetter(["x", "x", "y", "y"], "z"), "x")
