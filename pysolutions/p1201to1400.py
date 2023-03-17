from bisect import bisect_left
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

    def findTheDistanceValue1(self, arr1: List[int], arr2: List[int], d: int) -> int:
        # 1385.Find the Distance Value Between Two Arrays
        check = []
        for i in range(-d, d + 1):
            check.extend(map(lambda x: x + i, arr2))
        ans = 0
        for num in arr1:
            if num not in check:
                ans += 1
        return ans

    def findTheDistanceValue2(self, arr1: List[int], arr2: List[int], d: int) -> int:
        # 1385.Find the Distance Value Between Two Arrays
        # Solution2: Using bisect_left
        arr2.sort()
        n = len(arr2)
        count = 0
        for x in arr1:
            i = bisect_left(arr2, x)
            if (i == n or arr2[i] - x > d) and (i == 0 or x - arr2[i - 1] > d):
                count += 1
        return count
