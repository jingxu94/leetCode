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

    def test_makeConnected(self):
        # 1319.Number of Operations to Make Network Connected
        self.assertEqual(self.sl.makeConnected(4, [[0, 1], [0, 2], [1, 2]]), 1)
        self.assertEqual(self.sl.makeConnected(6, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]), 2)
        self.assertEqual(self.sl.makeConnected(6, [[0, 1], [0, 2], [0, 3], [1, 2]]), -1)
        connections = [
            [17, 51],
            [33, 83],
            [53, 62],
            [25, 34],
            [35, 90],
            [29, 41],
            [14, 53],
            [40, 84],
            [41, 64],
            [13, 68],
            [44, 85],
            [57, 58],
            [50, 74],
            [20, 69],
            [15, 62],
            [25, 88],
            [4, 56],
            [37, 39],
            [30, 62],
            [69, 79],
            [33, 85],
            [24, 83],
            [35, 77],
            [2, 73],
            [6, 28],
            [46, 98],
            [11, 82],
            [29, 72],
            [67, 71],
            [12, 49],
            [42, 56],
            [56, 65],
            [40, 70],
            [24, 64],
            [29, 51],
            [20, 27],
            [45, 88],
            [58, 92],
            [60, 99],
            [33, 46],
            [19, 69],
            [33, 89],
            [54, 82],
            [16, 50],
            [35, 73],
            [19, 45],
            [19, 72],
            [1, 79],
            [27, 80],
            [22, 41],
            [52, 61],
            [50, 85],
            [27, 45],
            [4, 84],
            [11, 96],
            [0, 99],
            [29, 94],
            [9, 19],
            [66, 99],
            [20, 39],
            [16, 85],
            [12, 27],
            [16, 67],
            [61, 80],
            [67, 83],
            [16, 17],
            [24, 27],
            [16, 25],
            [41, 79],
            [51, 95],
            [46, 47],
            [27, 51],
            [31, 44],
            [0, 69],
            [61, 63],
            [33, 95],
            [17, 88],
            [70, 87],
            [40, 42],
            [21, 42],
            [67, 77],
            [33, 65],
            [3, 25],
            [39, 83],
            [34, 40],
            [15, 79],
            [30, 90],
            [58, 95],
            [45, 56],
            [37, 48],
            [24, 91],
            [31, 93],
            [83, 90],
            [17, 86],
            [61, 65],
            [15, 48],
            [34, 56],
            [12, 26],
            [39, 98],
            [1, 48],
            [21, 76],
            [72, 96],
            [30, 69],
            [46, 80],
            [6, 29],
            [29, 81],
            [22, 77],
            [85, 90],
            [79, 83],
            [6, 26],
            [33, 57],
            [3, 65],
            [63, 84],
            [77, 94],
            [26, 90],
            [64, 77],
            [0, 3],
            [27, 97],
            [66, 89],
            [18, 77],
            [27, 43],
        ]
        self.assertEqual(self.sl.makeConnected(100, connections), 13)

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
