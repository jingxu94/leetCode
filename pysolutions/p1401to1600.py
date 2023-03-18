from typing import List


class Pro1401To1600:
    def __init__(self):
        pass

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


class BrowserHistory:
    # 1472.Design Browser History
    def __init__(self, homepage: str):
        self._history, self._future = [], []
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
