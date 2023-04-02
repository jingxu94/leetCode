from collections import Counter
from typing import List


class Pro1801To2000:
    def __init__(self) -> None:
        pass

    def arraySign(self, nums: List[int]) -> int:
        # 1822.Sign of the Product of an Array
        if 0 in nums:
            return 0
        ct_num = Counter(nums)
        under_zero = 0
        for key in ct_num.keys():
            if key < 0:
                under_zero += ct_num[key]
        if under_zero % 2 == 1:
            return -1
        return 1

    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        # 1855.Maximum Distance Between a Pair of Values
        i, j = 0, 0
        max_dist = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                max_dist = max(max_dist, j - i)
                j += 1
            else:
                i += 1
                j = max(j, i)
        return max_dist

    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        # 1886.Determine Whether Matrix Can Be Obtained By Rotation
        if mat == target:
            return True

        def rotate(mat: List[List[int]]):
            return [list(x) for x in zip(*mat)][::-1]

        for _ in range(3):
            mat = rotate(mat)
            if mat == target:
                return True
        return False
