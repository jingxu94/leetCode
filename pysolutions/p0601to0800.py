from typing import List


class Pro0601To0800:
    def __init__(self):
        pass

    def search(self, nums: List[int], target: int) -> int:
        # 704.Binary Search
        def bin_search(nums: List[int], target: int, base: int):
            if len(nums) == 0:
                return -1
            indmid = len(nums) // 2
            if target == nums[indmid]:
                return base + indmid
            elif target > nums[indmid]:
                return bin_search(nums[indmid + 1 :], target, base + indmid + 1)
            else:
                return bin_search(nums[:indmid], target, base)

        return bin_search(nums, target, 0)

    def pivotIndex(self, nums: List[int]) -> int:
        # 724.Find Pivot Index
        total = sum(nums)
        lsum = 0
        for i in range(len(nums)):
            if total - nums[i] - lsum == lsum:
                return i
            lsum += nums[i]
        return -1

    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        # 744.Find Smallest Letter Greater Than Target
        if letters[-1] <= target:
            return letters[0]
        left, right = 0, len(letters) - 1
        while left < right:
            mid = (left + right) // 2
            if letters[mid] > target:
                right = mid
            elif letters[mid] <= target:
                left = mid + 1
        return letters[left]
