import random
from collections import Counter
from typing import List, Optional

from .utils import ListNode


class Pro0201To0400:
    def __init__(self):
        pass

    def containsDuplicate(self, nums: List[int]) -> bool:
        # 217.Contains Duplicate
        cont = Counter(nums)
        for key in cont.elements():
            if cont[key] > 1:
                return True
        return False

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # 234.Palindrome Linked List
        fd_mid, fd_end = head, head
        tmp_reverse = None
        while fd_end and fd_end.next:
            fd_mid = fd_mid.next
            fd_end = fd_end.next.next
        tmp_reverse, fd_mid, tmp_reverse.next = fd_mid, fd_mid.next, None
        while fd_mid:
            fd_mid.next, tmp_reverse, fd_mid = tmp_reverse, fd_mid, fd_mid.next
        fd_end, fd_mid = head, tmp_reverse
        while fd_mid:
            if fd_end.val != fd_mid.val:
                return False
            fd_end, fd_mid = fd_end.next, fd_mid.next
        return True

    def firstBadVersion(self, n: int) -> int:
        # 278.First Bad Version
        left, right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left

    def moveZeroes(self, nums: List[int]) -> List[int]:
        """283.Move Zeroes
        Do not return anything, modify nums in-place instead.
        """
        nactivate = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[nactivate] = nums[i]
                nactivate += 1
        for i in range(nactivate, len(nums)):
            nums[i] = 0

        return nums

    def guessNumber(self, n: int) -> int:
        # 374.Guess Number Higher or Lower
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            callback = guess(mid)
            if callback == 0:
                return mid
            elif callback < 0:
                right = mid - 1
            else:
                left = mid + 1

        return -1

    def getRandom(self, head: Optional[ListNode]) -> int:
        # 382.Linked List Random Node
        maxnode = 1
        result = 0
        curr = head
        while curr:
            if random.random() < 1 / maxnode:
                result = curr.val
            curr = curr.next
            maxnode += 1
        return result

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # 383.Ransom Note
        note, mag = Counter(ransomNote), Counter(magazine)
        return (note & mag) == note
