from collections import defaultdict
from typing import List


class Pro1401To1600:
    def __init__(self):
        pass

    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        # 1466.Reorder Routes to Make All Paths Lead to the City Zero
        def dfs(node):
            checked.add(node)
            ans = 0
            for neighbor, need_reversed in roads[node]:
                if neighbor not in checked:
                    ans += need_reversed
                    ans += dfs(neighbor)
            return ans

        roads = defaultdict(list)
        for a, b in connections:
            roads[a].append((b, 1))  # Edge from a to b needs to be reversed
            roads[b].append((a, 0))  # Edge from b to a is already in the correct direction
        checked: set[int] = set()
        return dfs(0)

    def runningSum(self, nums: List[int]) -> List[int]:
        # 1480.Running Sum of 1d Array
        ans = nums
        if len(nums) > 1:
            for i in range(1, len(nums)):
                ans[i] = ans[i - 1] + nums[i]
        return ans

    def average(self, salary: List[int]) -> float:
        # 1491.Average Salary Excluding the Minimum and Maximum Salary
        return (sum(salary) - max(salary) - min(salary)) / (len(salary) - 2)

    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        # 1502.Can Make Arithmetic Progression From Sequence
        length = len(arr)
        if length == 2:
            return True
        arr.sort()
        gap = arr[0] - arr[1]
        for i in range(1, length - 1):
            if arr[i] - arr[i + 1] != gap:
                return False
        return True

    def countOdds(self, low: int, high: int) -> int:
        # 1523.Count Odd Numbers in an Interval Range
        nodd = (high - low) // 2
        if low % 2 or high % 2:
            nodd += 1
        return nodd

    def findKthPositive(self, arr: List[int], k: int) -> int:
        # 1539.Kth Missing Positive Number
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] - (mid + 1) < k:
                left = mid + 1
            else:
                right = mid
        return left + k

    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        # 1546.Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
        seen = set([0])
        ans = curr = 0
        for num in nums:
            curr += num
            prev = curr - target
            if prev in seen:
                ans += 1
                seen = set()
            seen.add(curr)
        return ans

    def diagonalSum(self, mat: List[List[int]]) -> int:
        # 1572.Matrix Diagonal Sum
        rcs = len(mat)
        pdiag, sdiag = 0, 0
        for i in range(rcs):
            pdiag += mat[i][i]
            sdiag += mat[i][rcs - 1 - i]
        if rcs % 2 == 1:
            mid = rcs // 2
            sdiag -= mat[mid][mid]
        return pdiag + sdiag

    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        # 1588.Sum of All Odd Length Subarrays
        total_sum = 0
        n = len(arr)
        for i in range(n):
            left_count = i
            right_count = n - i - 1
            left_odd = (left_count + 1) // 2
            right_odd = (right_count + 1) // 2
            left_even = left_count // 2 + 1
            right_even = right_count // 2 + 1
            total_sum += arr[i] * (left_odd * right_odd + left_even * right_even)
        return total_sum


class BrowserHistory:  # pragma: no cover
    # 1472.Design Browser History
    def __init__(self, homepage: str):
        self._history: List[str] = []
        self._future: List[str] = []
        self._curr = homepage

    def visit(self, url: str) -> None:
        self._history.append(self._curr)
        self._curr = url
        self._future = []

    def back(self, steps: int) -> str:
        while steps > 0 and self._history:
            self._future.append(self._curr)
            self._curr = self._history.pop()
            steps -= 1
        return self._curr

    def forward(self, steps: int) -> str:
        while steps > 0 and self._future:
            self._history.append(self._curr)
            self._curr = self._future.pop()
            steps -= 1
        return self._curr
