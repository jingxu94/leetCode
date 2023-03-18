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
