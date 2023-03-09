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

    def longestCommonPrefix(self, strs: List[str]) -> str:
        # 14.Longest Common Prefix
        prefix = []
        for i in range(len(strs[0])):
            for alpha in strs[1:]:
                try:
                    if strs[0][i] != alpha[i]:
                        return "".join(prefix)
                except IndexError:
                    return "".join(prefix)
            prefix.append(strs[0][i])
        return strs[0]

    def isValid(self, s: str) -> bool:
        # 20.Valid Parentheses
        stack = []
        for i in range(len(s)):
            if stack == [] and s[i] in (")", "]", "}"):
                return False
            elif s[i] in ("(", "[", "{"):
                stack.append(s[i])
            elif stack[-1] == "(" and s[i] == ")":
                stack.pop()
            elif stack[-1] == "[" and s[i] == "]":
                stack.pop()
            elif stack[-1] == "{" and s[i] == "}":
                stack.pop()
            elif s[i] in (")", "]", "}"):
                return False
        return stack == []

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 21.Merge Two Sorted Lists
        result = ListNode()
        curr = result
        while list1 and list2:
            if list1.val < list2.val:
                curr.next = ListNode(list1.val)
                curr, list1 = curr.next, list1.next
            else:
                curr.next = ListNode(list2.val)
                curr, list2 = curr.next, list2.next
        if list1:
            curr.next = list1
        if list2:
            curr.next = list2
        return result.next

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 142.Linked List Cycle II
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None
        pos1, pos2 = head, slow
        while pos1 != pos2:
            pos1 = pos1.next
            pos2 = pos2.next
        return pos1

    def removeDuplicates(self, nums: List[int]) -> List[int]:
        # 26.Remove Duplicates from Sorted Array
        nums[:] = sorted(frozenset(nums))
        # return len(nums)
        return nums
