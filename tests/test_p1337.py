import unittest

from P1337_K_Weakest_Rows_Matrix import Solution


class TestKWRM(unittest.TestCase):
    def test_caseone(self):
        mat = [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
        k = 3
        result = [2, 0, 3]
        self.assertEqual(result, Solution().kWeakestRows(mat, k))

    def test_casetwo(self):
        mat = [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0]]
        k = 2
        result = [0, 2]
        self.assertEqual(result, Solution().kWeakestRows(mat, k))
