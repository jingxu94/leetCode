from bisect import bisect_left
from typing import List


class Pro1201To1400:
    def __init__(self):
        pass

    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        """1232.Check If It Is a Straight Line
        Check if the given coordinates form a straight line.

        Args:
        coordinates (List[List[int]]): List of x, y coordinates.

        Returns:
        bool: True if coordinates form a straight line, False otherwise.
        """
        if len(coordinates) < 3:
            return True
        dx, dy = coordinates[0][0] - coordinates[1][0], coordinates[0][1] - coordinates[1][1]
        for point in coordinates[2:]:
            if (coordinates[0][0] - point[0]) * dy != (coordinates[0][1] - point[1]) * dx:
                return False
        return True

    def subtractProductAndSum(self, n: int) -> int:
        """1281.Subtract the Product and Sum of Digits of an Integer
        Subtract the product of digits and the sum of digits of an integer.

        Args:
        n (int): The integer.

        Returns:
        int: The difference between the product and sum of the digits of the integer.
        """
        digits = str(n)
        pd, sm = 1, 0
        for i in range(len(digits)):
            pd *= int(digits[i])
            sm += int(digits[i])
        return pd - sm

    def maximum69Number(self, num: int) -> int:
        """1323.Maximum 69 Number
        Change the first 6 in the given number to 9 and return the resulting number.

        Args:
        num (int): The input number.

        Returns:
        int: The maximum number that can be formed by changing the first 6 to 9.
        """
        numstr = str(num)
        if "6" not in numstr:
            return num
        return num + 3 * 10 ** (len(numstr) - numstr.index("6") - 1)

    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        """1337.The K Weakest Rows in a Matrix
        Find the K weakest rows in a matrix based on the number of soldiers.

        Args:
        mat (List[List[int]]): The input matrix.
        k (int): The number of weakest rows to return.

        Returns:
        List[int]: List of indices of the K weakest rows.
        """
        soldiers = []
        for row in range(len(mat)):
            soldiers.append(sum(mat[row]))
        arg_soldiers = sorted(range(len(soldiers)), key=soldiers.__getitem__)
        return arg_soldiers[:k]

    def numberOfSteps(self, num: int) -> int:
        """1342.Number of Steps to Reduce a Number to Zero
        Calculate the number of steps to reduce a number to zero by dividing by 2 if it's even or subtracting 1 if it's odd.

        Args:
        num (int): The input number.

        Returns:
        int: The number of steps to reduce the number to zero.
        """
        steps = 0
        while num > 0:
            if num % 2 == 0:
                num = num // 2
            else:
                num -= 1
            steps += 1
        return steps

    def countNegatives(self, grid: List[List[int]]) -> int:
        # 1351.Count Negative Numbers in a Sorted Matrix
        def binary_search(row: List[int]) -> int:
            left, right = 0, len(row)
            while left < right:
                mid = (left + right) // 2
                if row[mid] < 0:
                    right = mid
                else:
                    left = mid + 1
            return len(row) - left

        count = 0
        for row in grid:
            count += binary_search(row)
        return count

    def findTheDistanceValue1(self, arr1: List[int], arr2: List[int], d: int) -> int:
        """1385.Find the Distance Value Between Two Arrays
        Find the distance value between two arrays.

        Args:
        arr1 (List[int]): The first array.
        arr2 (List[int]): The second array.
        d (int): The distance value.

        Returns:
        int: The distance value between two arrays.
        """
        check = []
        for i in range(-d, d + 1):
            check.extend(map(lambda x: x + i, arr2))
        ans = 0
        for num in arr1:
            if num not in check:
                ans += 1
        return ans

    def findTheDistanceValue2(self, arr1: List[int], arr2: List[int], d: int) -> int:
        """1385.Find the Distance Value Between Two Arrays
        Find the distance value between two arrays using bisect_left.

        Args:
        arr1 (List[int]): The first array.
        arr2 (List[int]): The second array.
        d (int): The distance value.

        Returns:
        int: The distance value between two arrays.
        """
        # Solution2: Using bisect_left
        arr2.sort()
        n = len(arr2)
        count = 0
        for x in arr1:
            i = bisect_left(arr2, x)
            if (i == n or arr2[i] - x > d) and (i == 0 or x - arr2[i - 1] > d):
                count += 1
        return count
