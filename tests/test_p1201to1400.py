import unittest

from pysolutions import Pro1201To1400


class TestP1201To1400(unittest.TestCase):
    @property
    def sl(self):
        return Pro1201To1400()

    def test_checkStraightLine(self):
        # 1232.Check If It Is a Straight Line
        self.assertTrue(self.sl.checkStraightLine([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]))
        self.assertFalse(self.sl.checkStraightLine([[1, 1], [2, 2], [3, 4], [4, 5], [5, 6], [7, 7]]))
        self.assertFalse(self.sl.checkStraightLine([[1, 2], [2, 3], [3, 5]]))

    def test_subtractProductAndSum(self):
        # 1281.Subtract the Product and Sum of Digits of an Integer
        self.assertEqual(self.sl.subtractProductAndSum(234), 15)
        self.assertEqual(self.sl.subtractProductAndSum(4421), 21)

    def test_maximum69Number(self):
        # 1323.Maximum 69 Number
        self.assertEqual(self.sl.maximum69Number(9669), 9969)
        self.assertEqual(self.sl.maximum69Number(9996), 9999)
        self.assertEqual(self.sl.maximum69Number(9999), 9999)

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

    def test_countNegatives(self):
        # 1351.Count Negative Numbers in a Sorted Matrix
        self.assertEqual(self.sl.countNegatives([[4, 3, 2, -1], [3, 2, 1, -1], [1, 1, -1, -2], [-1, -1, -2, -3]]), 8)
        self.assertEqual(self.sl.countNegatives([[3, 2], [1, 0]]), 0)

    def test_findTheDistanceValue(self):
        # 1385.Find the Distance Value Between Two Arrays
        arr1, arr2, d, ans = [4, 5, 8], [10, 9, 1, 8], 2, 2
        self.assertEqual(self.sl.findTheDistanceValue1(arr1, arr2, d), ans)
        self.assertEqual(self.sl.findTheDistanceValue2(arr1, arr2, d), ans)
        arr1, arr2, d, ans = [1, 4, 2, 3], [-4, -3, 6, 10, 20, 30], 3, 2
        self.assertEqual(self.sl.findTheDistanceValue1(arr1, arr2, d), ans)
        self.assertEqual(self.sl.findTheDistanceValue2(arr1, arr2, d), ans)
        arr1, arr2, d, ans = [2, 1, 100, 3], [-5, -2, 10, -3, 7], 6, 1
        self.assertEqual(self.sl.findTheDistanceValue1(arr1, arr2, d), ans)
        self.assertEqual(self.sl.findTheDistanceValue2(arr1, arr2, d), ans)
