from typing import List, Optional

from .utils import ListNode


class Pro0001To0200:
    def __init__(self):
        pass

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 1.Two Sum
        checked = {}
        i = 0
        while target - nums[i] not in checked:
            checked[nums[i]] = i
            i += 1

        return [checked[target - nums[i]], i]

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 2.Add Two Numberes
        result = ListNode()
        curr = result
        over_ten = 0
        while l1 or l2 or over_ten:
            value = over_ten
            if l1:
                value += l1.val
                l1 = l1.next
            if l2:
                value += l2.val
                l2 = l2.next
            over_ten, value = divmod(value, 10)
            curr.next = ListNode(value)
            curr = curr.next
        return result.next

    def isPalindrome(self, x: int) -> bool:
        # 9.Palindrome Number
        if x < 0:
            return False
        x_raw = x
        x_flip = 0
        while x // 10 > 0:
            x_flip = x_flip * 10 + (x % 10)
            x = x // 10
        x_flip = x_flip * 10 + (x % 10)
        return x_raw == x_flip

    def romanToInt(self, s: str) -> int:
        """13.Roman to Integer
        Roman numerals are represented by seven different symbols:
        I, V, X, L, C, D and M.
        -----------------------
        Symbol    Value
          I         1
          V         5
          X         10
          L         50
          C         100
          D         500
          M         1000
        -----------------------
        """
        roman = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }
        result = roman[s[-1]]
        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i + 1]]:
                result -= roman[s[i]]
            else:
                result += roman[s[i]]
        return result
