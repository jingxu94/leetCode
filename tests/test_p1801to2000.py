import unittest

from pysolutions import Pro1801To2000


class TestP1801To2000(unittest.TestCase):
    @property
    def sl(self):
        return Pro1801To2000()

    def test_maxValue(self):
        # 1802.Maximum Value at a Given Index in a Bounded Array
        self.assertEqual(self.sl.maxValue(4, 2, 6), 2)
        self.assertEqual(self.sl.maxValue(6, 1, 10), 3)

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

    def test_minSpeedOnTime(self):
        # 1870.Minimum Speed to Arrive on Time
        self.assertEqual(self.sl.minSpeedOnTime([1, 3, 2], 6), 1)
        self.assertEqual(self.sl.minSpeedOnTime([1, 3, 2], 2.7), 3)
        self.assertEqual(self.sl.minSpeedOnTime([1, 3, 2], 1.9), -1)

    def test_findRotation(self):
        # 1886.Determine Whether Matrix Can Be Obtained By Rotation
        self.assertTrue(self.sl.findRotation([[0, 1], [1, 0]], [[0, 1], [1, 0]]))
        self.assertTrue(self.sl.findRotation([[0, 1], [1, 0]], [[1, 0], [0, 1]]))
        self.assertFalse(self.sl.findRotation([[0, 1], [1, 1]], [[1, 0], [0, 1]]))

    def test_nearestExit(self):
        # 1926.Nearest Exit from Entrance in Maze
        maze = [["+", "+", ".", "+"], [".", ".", ".", "+"], ["+", "+", "+", "."]]
        self.assertEqual(self.sl.nearestExit(maze, [1, 2]), 1)
        maze = [["+", "+", "+"], [".", ".", "."], ["+", "+", "+"]]
        self.assertEqual(self.sl.nearestExit(maze, [1, 0]), 2)
        maze = [[".", "+"]]
        self.assertEqual(self.sl.nearestExit(maze, [0, 0]), -1)
        maze = [
            ["+", ".", "+", "+", "+", "+", "+"],
            ["+", ".", "+", ".", ".", ".", "+"],
            ["+", ".", "+", ".", "+", ".", "+"],
            ["+", ".", ".", ".", "+", ".", "+"],
            ["+", "+", "+", "+", "+", "+", "."],
        ]
        self.assertEqual(self.sl.nearestExit(maze, [0, 1]), -1)

    def test_longestObstacleCourseAtEachPosition(self):
        # 1964.Find the Longest Valid Obstacle Course at Each Position
        self.assertEqual(self.sl.longestObstacleCourseAtEachPosition([1, 2, 3, 2]), [1, 2, 3, 3])
        self.assertEqual(self.sl.longestObstacleCourseAtEachPosition([2, 2, 1]), [1, 2, 1])
        self.assertEqual(self.sl.longestObstacleCourseAtEachPosition([3, 1, 5, 6, 4, 2]), [1, 1, 2, 3, 2, 2])

    def test_latestDayToCross(self):
        # 1970.Last Day Where You Can Still Cross
        self.assertEqual(self.sl.latestDayToCross(0, 0, []), -1)
        self.assertEqual(self.sl.latestDayToCross(2, 2, [[1, 1], [2, 1], [1, 2], [2, 2]]), 2)
        self.assertEqual(self.sl.latestDayToCross(2, 2, [[1, 1], [1, 2], [2, 1], [2, 2]]), 1)
        self.assertEqual(
            self.sl.latestDayToCross(3, 3, [[1, 2], [2, 1], [3, 3], [2, 2], [1, 1], [1, 3], [2, 3], [3, 2], [3, 1]]), 3
        )
