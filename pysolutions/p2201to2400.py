from typing import List


class Pro2201To2400:
    def __init__(self) -> None:
        pass

    def zeroFilledSubarray(self, nums: List[int]) -> int:
        # 2348.Number of Zero-Filled Subarrays
        zero_subs = []
        lf = rf = -1
        for i, num in enumerate(nums):
            if num == 0 and lf == -1 and rf == -1:
                lf = rf = i
            elif num == 0:
                rf = i
            elif num != 0 and rf != -1:
                zero_subs.append(rf - lf + 1)
                lf = rf = -1
            if i == len(nums) - 1 and num == 0:
                zero_subs.append(rf - lf + 1)
        ans = 0
        for nzero in zero_subs:
            ans += ((1 + nzero) * nzero) // 2
        return ans
