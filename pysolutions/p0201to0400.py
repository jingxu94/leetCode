import random
from collections import Counter
from typing import List, Optional

from .utils import ListNode


class Pro0201To0400:
    def __init__(self):
        pass

    def isIsomorphic(self, s: str, t: str) -> bool:
        # 205.Isomoriphic Strings
        if len(s) != len(t):
            return False
        mapping_st, mapping_ts = dict(), dict()
        for i in range(len(s)):
            if mapping_ts.get(t[i]) is None:
                mapping_ts[t[i]] = s[i]
            if mapping_st.get(s[i]) is None:
                mapping_st[s[i]] = t[i]
            if s[i] != mapping_ts[t[i]] or t[i] != mapping_st[s[i]]:
                return False
        return True

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

    def reverseString(self, s: List[str]) -> None:
        """344.Reverse String
        Do not return anything, modify s in-place instead.
        """
        s.reverse()

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 350.Intersection of Two Arrays 2
        cnums1 = Counter(nums1)
        cnums2 = Counter(nums2)
        for key in cnums1.keys():
            num = cnums2.get(key, 0)
            cnums1[key] = min(num, cnums1[key])
        return list(cnums1.elements())

    def isPerfectSquare(self, num: int) -> bool:
        # 367.Valid Perfect Square
        left, right = 1, num
        while left <= right:
            mid = (left + right) // 2
            square = mid**2
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        return False

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

    def isSubsequence(self, s: str, t: str) -> bool:
        # 392.Is Subsequence
        if s == "":
            return True
        for i in range(len(t)):
            if s[0] == t[i]:
                return self.isSubsequence(s[1:], t[i + 1 :])
        return False


# 208.Implement Trie (Prefix Tree)
# class Trie:
#     def __init__(self):
#         self.words = []
#
#     def insert(self, word: str) -> None:
#         self.words.append(word)
#
#     def search(self, word: str) -> bool:
#         if word in self.words:
#             return True
#         return False
#
#     def startsWith(self, prefix: str) -> bool:
#         for word in self.words:
#             if word.startswith(prefix):
#                 return True
#         return False
