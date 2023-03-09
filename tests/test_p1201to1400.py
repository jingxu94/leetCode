import unittest

from pysolutions import Pro1201To1400


class TestP1201To1400(unittest.TestCase):
    @property
    def sl(self):
        return Pro1201To1400()

    def test_kWeakestRows(self):
        # 1337.The K Weakest Rows in a Matrix
        mat1 = [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
        k1 = 3
        res1 = [2, 0, 3]
        self.assertEqual(res1, self.sl.kWeakestRows(mat1, k1))

        mat2 = [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0]]
        k2 = 2
        res2 = [0, 2]
        self.assertEqual(res2, self.sl.kWeakestRows(mat2, k2))

    def test_numberOfSteps(self):
        # 1342.Number of Steps to Reduce a Number to Zero
        num1, num2, num3 = 14, 8, 123
        res1, res2, res3 = 6, 4, 12
        self.assertEqual(res1, self.sl.numberOfSteps(num1))
        self.assertEqual(res2, self.sl.numberOfSteps(num2))
        self.assertEqual(res3, self.sl.numberOfSteps(num3))
