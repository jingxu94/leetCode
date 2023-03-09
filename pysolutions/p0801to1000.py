from collections import Counter
from typing import List, Optional

from .utils import ListNode


class Pro0801To1000:
    def __init__(self):
        pass

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 875.Koko Eating Bananas
        def eat(piles: List[int], h: int, speed: int):
            time_list = []
            for pile in piles:
                time = pile // speed
                if pile % speed:
                    time += 1
                time_list.append(time)
            return sum(time_list) <= h

        left, right = 1, max(piles)
        result = -1
        while left <= right:
            speed = int((left + right) / 2)
            if eat(piles, h, speed):
                right = speed - 1
                result = speed
            else:
                left = speed + 1
        return result

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 876.Middle of the Linked List
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def sortArray(self, nums: List[int]) -> List[int]:
        # 912.Sort an Array
        nums_dict = Counter(nums)
        nums_sorted = []
        for num in range(min(nums), max(nums) + 1):
            nums_sorted.extend([num] * nums_dict[num])
        return nums_sorted
