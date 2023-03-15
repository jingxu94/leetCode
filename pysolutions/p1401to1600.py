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

    def countOdds(self, low: int, high: int) -> int:
        # 1523.Count Odd Numbers in an Interval Range
        nodd = (high - low) // 2
        if low % 2 or high % 2:
            nodd += 1
        return nodd
