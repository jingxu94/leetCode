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

    def test_findTheWinner(self):
        # 1823.Find the Winner of the Circular Game
        self.assertEqual(self.sl.findTheWinner(5, 2), 3)
        self.assertEqual(self.sl.findTheWinner(6, 5), 1)

    def test_maxDistance(self):
        # 1855.Maximum Distance Between a Pair of Values
        self.assertEqual(self.sl.maxDistance([55, 30, 5, 4, 2], [100, 20, 10, 10, 5]), 2)
        self.assertEqual(self.sl.maxDistance([2, 2, 2], [10, 10, 1]), 1)
        self.assertEqual(self.sl.maxDistance([30, 29, 19, 5], [25, 25, 25, 25, 25]), 2)

    def test_largestPathValue(self):
        # 1857.Largest Color Value in a Directed Graph
        self.assertEqual(self.sl.largestPathValue("abaca", [[0, 1], [0, 2], [2, 3], [3, 4]]), 3)
        self.assertEqual(
            self.sl.largestPathValue(
                "hhqhuqhqff",
                [
                    [0, 1],
                    [0, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [2, 7],
                    [6, 7],
                    [7, 8],
                    [3, 8],
                    [5, 8],
                    [8, 9],
                    [3, 9],
                    [6, 9],
                ],
            ),
            3,
        )
        self.assertEqual(self.sl.largestPathValue("a", [[0, 0]]), -1)

    def test_findRotation(self):
        self.assertTrue(self.sl.findRotation([[0, 1], [1, 0]], [[0, 1], [1, 0]]))
        self.assertTrue(self.sl.findRotation([[0, 1], [1, 0]], [[1, 0], [0, 1]]))
        self.assertFalse(self.sl.findRotation([[0, 1], [1, 1]], [[1, 0], [0, 1]]))
