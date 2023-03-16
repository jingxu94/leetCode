from typing import List


class Pro1201To1400:
    def __init__(self):
        pass

    def subtractProductAndSum(self, n: int) -> int:
        # 1281.Subtract the Product and Sum of Digits of an Integer
        digits = str(n)
        pd, sm = 1, 0
        for i in range(len(digits)):
            pd *= int(digits[i])
            sm += int(digits[i])
        return pd - sm

    def maximum69Number(self, num: int) -> int:
        # 1323.Maximum 69 Number
        numstr = str(num)
        if "6" not in numstr:
            return num
        return num + 3 * 10 ** (len(numstr) - numstr.index("6") - 1)

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
