class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        checked = {}
        i = 0
        while target - nums[i] not in checked:
            checked[nums[i]] = i
            i += 1

        return [checked[target - nums[i]], i]
