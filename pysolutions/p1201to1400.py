from typing import List


class Pro1201To1400:
    def __init__(self):
        pass

    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # 1337.The K Weakest Rows in a Matrix
        soldiers = []
        for row in range(len(mat)):
            soldiers.append(sum(mat[row]))
        arg_soldiers = sorted(range(len(soldiers)), key=soldiers.__getitem__)
        return arg_soldiers[:k]

    def numberOfSteps(self, num: int) -> int:
        # 1342.Number of Steps to Reduce a Number to Zero
        steps = 0
        while num > 0:
            if num % 2 == 0:
                num = num // 2
            else:
                num -= 1
            steps += 1
        return steps
