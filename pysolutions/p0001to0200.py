import math
import re
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import chain, combinations, permutations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .utils import ListNode, Node, TreeNode


class NodeTwo:  # pragma: no cover
    def __init__(self, x: int, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


class Pro0001To0200:
    def __init__(self):
        pass

    def twoSum_1(self, nums: List[int], target: int) -> List[int]:
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

    def lengthOfLongestSubstring(self, s: str) -> int:
        # 3.Longest Subtring Without Repeating Characters
        chars: Counter = Counter()
        left = right = 0
        ans = 0
        while right < len(s):
            rchar = s[right]
            chars[rchar] += 1
            while chars[rchar] > 1:
                lchar = s[left]
                chars[lchar] -= 1
                left += 1
            ans = max(ans, right - left + 1)
            right += 1
        return ans

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 4.Median of Two Sorted Arrays
        nums = sorted(nums1 + nums2)
        n = len(nums)
        if n % 2 == 0:
            return (nums[n // 2 - 1] + nums[n // 2]) / 2
        else:
            return nums[n // 2]

    def longestPalindrome(self, s: str) -> str:
        # 5.Longest Palindromic Substring
        m = n = len(s)
        while m > 1:
            mstrs = (s[i : i + m] for i in range(n - m + 1))
            for mstr in mstrs:
                if mstr == mstr[::-1]:
                    return mstr
            m -= 1
        return s[0]

    def convert(self, s: str, numRows: int) -> str:
        # 6.ZigZag Conversion
        if numRows == 1:
            return s
        rows = [""] * numRows
        i = 0
        while i < len(s):
            for j in range(numRows):
                if i < len(s):
                    rows[j] += s[i]
                    i += 1
            for j in range(numRows - 2, 0, -1):
                if i < len(s):
                    rows[j] += s[i]
                    i += 1
        return "".join(rows)

    def reverse(self, x: int) -> int:
        # 7.Reverse Integer
        if x < 0:
            return -self.reverse(-x)
        result = 0
        while x > 0:
            result = result * 10 + x % 10
            x = x // 10
        return result if result < 2**31 else 0

    def myAtoi(self, s: str) -> int:
        # 8.String to Integer (atoi)
        s = s.strip()
        if not s:
            return 0
        sign = 1
        if s[0] == "-":
            sign = -1
            s = s[1:]
        elif s[0] == "+":
            s = s[1:]
        result = 0
        for c in s:
            if c.isdigit():
                result = result * 10 + int(c)
            else:
                break
        result = result * sign
        return max(-(2**31), min(result, 2**31 - 1))

    def isPalindrome(self, x: int) -> bool:
        # 9.Palindrome Number
        if x < 0:
            return False
        div = 10 ** (len(str(x)) - 1)
        while x > 0:
            left = x // div
            right = x % 10
            if left != right:
                return False
            x = (x % div) // 10
            div = div // 100
        return True

    def isMatch(self, s: str, p: str) -> bool:
        # 10.Regular Expression Matching
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[0][j] = dp[0][j - 2]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == "." or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == "*":
                    dp[i][j] = dp[i][j - 2]
                    if p[j - 2] == "." or p[j - 2] == s[i - 1]:
                        dp[i][j] |= dp[i - 1][j]
        return dp[m][n]

    def maxArea(self, height: List[int]) -> int:
        # 11.Container With Most Water
        # n = len(height)
        # max_areas: List[int] = [0] * (n - 1)
        # for i in range(n - 1):
        #     for j in range(i + 1, n):
        #         max_areas[i] = max(max_areas[i], min(height[i], height[j]) * (j - i))
        # return max(max_areas)
        # ============================
        left, right = 0, len(height) - 1
        max_area = 0
        while left < right:
            area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area

    def intToRoman(self, num: int) -> str:
        # 12.Integer to Roman
        roman = {
            1: "I",
            4: "IV",
            5: "V",
            9: "IX",
            10: "X",
            40: "XL",
            50: "L",
            90: "XC",
            100: "C",
            400: "CD",
            500: "D",
            900: "CM",
            1000: "M",
        }
        result = ""
        for n in sorted(roman.keys(), reverse=True):
            while num >= n:
                result += roman[n]
                num -= n
        return result

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
        prefix: list = []
        for i in range(len(strs[0])):
            for alpha in strs[1:]:
                try:
                    if strs[0][i] != alpha[i]:
                        return "".join(prefix)
                except IndexError:
                    return "".join(prefix)
            prefix.append(strs[0][i])
        return strs[0]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 15.3Sum
        # permutation = (
        #     [nums[i], nums[j], nums[k]]
        #     for i in range(len(nums) - 2)
        #     for j in range(i + 1, len(nums) - 1)
        #     for k in range(j + 1, len(nums))
        #     if nums[i] + nums[j] + nums[k] == 0
        # )
        # unique_tuples = set(tuple(lst) for lst in permutation)
        # return [list(t) for t in unique_tuples]
        # =====================================
        ans: List[List[int]] = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
        return ans

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # 16.3Sum Closest
        nums.sort()
        ans = sum(nums[:3])
        for i in range(len(nums) - 2):
            left, right = i + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if abs(total - target) < abs(ans - target):
                    ans = total
                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    return ans
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        # 17.Letter Combinations of a Phone Number
        def dfs(digits, mapping, path, res):
            if not digits:
                res.append(path)
                return
            for char in mapping[digits[0]]:
                dfs(digits[1:], mapping, path + char, res)

        if not digits:
            return []
        mapping = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res: List[str] = []
        dfs(digits, mapping, "", res)
        return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 18.4Sum
        nums.sort()
        ans: List[List[int]] = []
        for i in range(len(nums) - 3):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j - 1] == nums[j]:
                    continue
                left, right = j + 1, len(nums) - 1
                while left < right:
                    total = nums[i] + nums[j] + nums[left] + nums[right]
                    if total < target:
                        left += 1
                    elif total > target:
                        right -= 1
                    else:
                        ans.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left - 1] == nums[left]:
                            left += 1
                        while left < right and nums[right + 1] == nums[right]:
                            right -= 1
        return ans

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 19.Remove Nth Node From End of List
        if head is None:
            return None
        length, curr = 0, head
        while curr:
            length, curr = length + 1, curr.next
        steps = length - n
        if steps == 0 and n == 1:
            return None
        elif steps == 0:
            return head.next
        else:
            curr = head
            for _ in range(steps - 1):
                curr = curr.next
            curr.next = curr.next.next
            return head

    def isValid(self, s: str) -> bool:
        # 20.Valid Parentheses
        stack: List[str] = []
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
            else:
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

    def generateParenthesis(self, n: int) -> List[str]:
        # 22.Generate Parentheses
        def backtrack(ans, cur, open_count, close_count, max_count):
            if len(cur) == max_count * 2:
                ans.append(cur)
                return
            if open_count < max_count:
                backtrack(ans, cur + "(", open_count + 1, close_count, max_count)
            if close_count < open_count:
                backtrack(ans, cur + ")", open_count, close_count + 1, max_count)

        ans: List[str] = []
        backtrack(ans, "", 0, 0, n)
        return ans

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 23.Merge k Sorted Lists
        # result = ListNode()
        # curr = result
        # nlist = len(lists)
        # if nlist == 0:
        #     return None
        # elif nlist == 1:
        #     return lists[0]
        # else:
        #     list1 = lists[0]
        #     for list2 in lists[1:]:
        #         while list1 and list2:
        #             if list1.val < list2.val:
        #                 curr.next = ListNode(list1.val)
        #                 curr, list1 = curr.next, list1.next
        #             else:
        #                 curr.next = ListNode(list2.val)
        #                 curr, list2 = curr.next, list2.next
        #         if list1:
        #             curr.next = list1
        #         if list2:
        #             curr.next = list2
        #         curr, list1 = result, result.next
        #     return result.next
        # =============================
        v = []
        for i in lists:
            x = i
            while x:
                v += [x.val]
                x = x.next
        v = sorted(v, reverse=True)
        ans = None
        for i in v:
            ans = ListNode(i, ans)
        return ans

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 24.Swap Nodes in Pairs
        if not head or not head.next:
            return head
        first = head
        second = head.next
        first.next = self.swapPairs(second.next)
        second.next = first
        return second

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 25.Reverse Nodes in k-Group
        if not head or not head.next:
            return head
        curr = head
        for _ in range(k):
            if not curr:
                return head
            curr = curr.next
        prev, curr = None, head
        for _ in range(k):
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        head.next = self.reverseKGroup(curr, k)
        return prev

    def removeDuplicates(self, nums: List[int]) -> List[int]:
        # 26.Remove Duplicates from Sorted Array
        nums[:] = sorted(frozenset(nums))
        # return len(nums)
        return nums

    def removeElement(self, nums: List[int], val: int) -> int:
        # 27.Remove Element
        while val in nums:
            nums.remove(val)
        return len(nums)

    def strStr(self, haystack: str, needle: str) -> int:
        # 28.Find the Index of the First Occurrence in a String
        index = 0
        while haystack:
            if haystack.startswith(needle):
                return index
            else:
                index += 1
                haystack = haystack[1:]
        return -1

    def divide(self, dividend: int, divisor: int) -> int:
        # 29.Divide Two Integers
        if dividend == -int(2**31) and divisor == -1:
            return int(2**31 - 1)
        if dividend == 0:
            return 0
        if divisor == 1:
            return dividend
        if divisor == -1:
            return -dividend
        sign = 1
        if dividend < 0:
            sign = -sign
            dividend = -dividend
        if divisor < 0:
            sign = -sign
            divisor = -divisor
        res = 0
        while dividend >= divisor:
            temp = divisor
            i = 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if sign == 1:
            return res
        else:
            return -res

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        # 30.Substring with Concatenation of All Words
        n = len(s)
        k = len(words)
        word_length = len(words[0])
        substring_size = word_length * k
        word_count = Counter(words)

        def sliding_window(left):
            words_found = defaultdict(int)
            words_used = 0
            excess_word = False
            # Do the same iteration pattern as the previous approach - iterate
            # word_length at a time, and at each iteration we focus on one word
            for right in range(left, n, word_length):
                if right + word_length > n:
                    break
                sub = s[right : right + word_length]
                if sub not in word_count:
                    # Mismatched word - reset the window
                    words_found = defaultdict(int)
                    words_used = 0
                    excess_word = False
                    left = right + word_length  # Retry at the next index
                else:
                    # If we reached max window size or have an excess word
                    while right - left == substring_size or excess_word:
                        # Move the left bound over continously
                        leftmost_word = s[left : left + word_length]
                        left += word_length
                        words_found[leftmost_word] -= 1
                        if words_found[leftmost_word] == word_count[leftmost_word]:
                            # This word was the excess word
                            excess_word = False
                        else:
                            # Otherwise we actually needed it
                            words_used -= 1
                    # Keep track of how many times this word occurs in the window
                    words_found[sub] += 1
                    if words_found[sub] <= word_count[sub]:
                        words_used += 1
                    else:
                        # Found too many instances already
                        excess_word = True

                    if words_used == k and not excess_word:
                        # Found a valid substring
                        answer.append(left)

        answer: List[int] = []
        for i in range(word_length):
            sliding_window(i)
        return answer

    def nextPermutation(self, nums: List[int]) -> None:  # pragma: no cover
        # 31.Next Permutation
        # Do not return anything, modify nums in-place instead.
        if len(nums) <= 1:
            return
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            right -= 1
            left += 1

    def longestValidParentheses(self, s: str) -> int:
        # 32.Longest Valid Parentheses
        stack = [-1]
        ans = 0
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            elif stack[-1] != -1 and s[stack[-1]] == "(":
                stack.pop()
                ans = max(ans, i - stack[-1])
            else:
                stack.append(i)
        return ans

    def search(self, nums: List[int], target: int) -> int:
        # 33.Search in Rotated Sorted Array
        if not nums:
            return -1
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            if nums[low] <= nums[mid]:
                if nums[low] <= target < nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] < target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 34.Find First and Last Position of Element in Sorted Array
        def find_left(nums, target):
            low, high = 0, len(nums) - 1
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] < target:
                    low = mid + 1
                else:
                    high = mid - 1
            return low

        def find_right(nums, target):
            low, high = 0, len(nums) - 1
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] <= target:
                    low = mid + 1
                else:
                    high = mid - 1
            return high

        left = find_left(nums, target)
        right = find_right(nums, target)

        if left <= right:
            return [left, right]
        else:
            return [-1, -1]

    def searchInsert(self, nums: List[int], target: int) -> int:
        # 35.Search Insert Position
        if target <= nums[0]:
            return 0
        for i in range(len(nums) - 1):
            if nums[i] < target <= nums[i + 1]:
                return i + 1
        return len(nums)

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 36.Valid Sudoku
        res = []
        for i in range(9):
            for j in range(9):
                element = board[i][j]
                if element != ".":
                    res += [(i, element), (element, j), (i // 3, j // 3, element)]
        return len(res) == len(set(res))

    def solveSudoku(self, board: List[List[str]]) -> None:  # pragma: no cover
        # 37.Sudoku Solver
        def is_valid(board, row, col, c):
            for i in range(9):
                if board[i][col] != "." and board[i][col] == c:
                    return False
                if board[row][i] != "." and board[row][i] == c:
                    return False
                if (
                    board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] != "."
                    and board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c
                ):
                    return False
            return True

        def backtrack(board, row, col):
            if col == 9:
                return backtrack(board, row + 1, 0)
            if row == 9:
                return True
            for i in range(row, 9):
                for j in range(col, 9):
                    if board[i][j] != ".":
                        return backtrack(board, i, j + 1)
                    for c in range(1, 10):
                        if not is_valid(board, i, j, str(c)):
                            continue
                        board[i][j] = str(c)
                        if backtrack(board, i, j + 1):
                            return True
                        board[i][j] = "."
                    return False
            return False

        backtrack(board, 0, 0)

    def countAndSay(self, n: int) -> str:
        # 38.Count and Say
        if n == 1:
            return "1"
        s = self.countAndSay(n - 1)
        res = ""
        count = 1
        for i in range(len(s)):
            if i == len(s) - 1 or s[i] != s[i + 1]:
                res += str(count) + s[i]
                count = 1
            else:
                count += 1
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 39.Combination Sum
        def backtrack(start, target, comb, ans):
            if target == 0:
                ans.append(list(comb))
                return
            elif target < 0:
                return
            for i in range(start, len(candidates)):
                comb.append(candidates[i])
                backtrack(i, target - candidates[i], comb, ans)
                comb.pop()

        ans: List[List[int]] = []
        backtrack(0, target, [], ans)
        return ans

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 40.Combination Sum II
        def backtrack(start, target, comb, ans):
            if target == 0:
                ans.append(list(comb))
                return
            elif target < 0:
                return
            for i in range(start, len(candidates)):
                # Skip duplicates
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                comb.append(candidates[i])
                backtrack(i + 1, target - candidates[i], comb, ans)
                comb.pop()

        ans: List[List[int]] = []
        candidates.sort()
        backtrack(0, target, [], ans)
        return ans

    def firstMissingPositive(self, nums: List[int]) -> int:
        # 41.First Missing Positive
        n = len(nums)
        for i in range(n):
            while 0 < nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                # Swap
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1

    def trap(self, height: List[int]) -> int:
        # 42.Trapping Rain Water
        left, right, up, ans = 0, len(height) - 1, 0, 0
        while left < right:
            low = min(height[left], height[right])
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
            up = max(up, low)
            ans += up - low
        return ans

    def multiply(self, num1: str, num2: str) -> str:
        # 43.Multiply Strings
        return str(eval(num1 + "*" + num2))

    def jump(self, nums: List[int]) -> int:
        # 45.Jump Game II
        # n = len(nums)
        # if n == 1:
        #     return 0
        # min_steps = [math.inf] * n
        # min_steps[0] = 0
        # for i, num in enumerate(nums[:-1]):
        #     for j in range(i + 1, i + num + 1):
        #         if j < n:
        #             min_steps[j] = min(min_steps[j], min_steps[i] + 1)
        # return int(min_steps[-1])
        # ================================
        n = len(nums)
        if n == 1:
            return 0
        jumps = 0
        curr_end = curr_farthest = 0
        for i in range(n - 1):
            curr_farthest = max(curr_farthest, i + nums[i])
            if i == curr_end:
                jumps += 1
                curr_end = curr_farthest
                if curr_end >= n - 1:
                    return jumps
        return -1

    def permute(self, nums: List[int]) -> List[List[int]]:
        # 46.Permutations
        def backtrack(path):
            if len(path) == len(nums):
                result.append(path[:])
                return
            for num in nums:
                if num not in path:
                    path.append(num)
                    backtrack(path)
                    path.pop()

        result: List[List[int]] = []
        backtrack([])
        return result

    def permuteUnique(self, nums: List[int]) -> Any:  # pragma: no cover
        # 47.Permutations II
        return list(set(permutations(nums, len(nums))))

    def rotate_v2(self, matrix: List[List[int]]) -> None:  # pragma: no cover
        # 48.Rotate Image
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 49.Group Anagrams
        strs_table: Dict[str, List[str]] = {}
        for word in strs:
            sorted_word = "".join(sorted(word))
            if sorted_word not in strs_table:
                strs_table[sorted_word] = []
            strs_table[sorted_word].append(word)
        return list(strs_table.values())

    def myPow(self, x: float, n: int) -> float:
        # 50.Pow(x,n)
        def function(base=x, exponent=abs(n)):
            if exponent == 0:
                return 1
            elif exponent % 2 == 0:
                return function(base * base, exponent // 2)
            else:
                return base * function(base * base, (exponent - 1) // 2)

        f = function()
        return float(f) if n >= 0 else 1 / f

    def binaryExp(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        # Handle case where, n < 0.
        if n < 0:
            n = -1 * n
            x = 1.0 / x
        # Perform Binary Exponentiation.
        result = 1.0
        while n != 0:
            # If 'n' is odd we multiply result with 'x' and reduce 'n' by '1'.
            if n % 2 == 1:
                result *= x
                n -= 1
            # We square 'x' and reduce 'n' by half, x^n => (x^2)^(n/2).
            x *= x
            n //= 2
        return result

    def myPow_v2(self, x: float, n: int) -> float:
        return self.binaryExp(x, n)

    def solveNQueens(self, n: int) -> List[List[str]]:
        # 51.N-Queens
        def backtrack(row=0):
            for col in range(n):
                if is_not_under_attack(row, col):
                    place_queen(row, col)
                    if row + 1 == n:
                        add_solution()
                    else:
                        backtrack(row + 1)
                    remove_queen(row, col)

        def is_not_under_attack(row, col):
            return not (cols[col] + hill_diagonals[row - col] + dale_diagonals[row + col])

        def place_queen(row, col):
            queens.add((row, col))
            cols[col] = 1
            hill_diagonals[row - col] = 1
            dale_diagonals[row + col] = 1

        def remove_queen(row, col):
            queens.remove((row, col))
            cols[col] = 0
            hill_diagonals[row - col] = 0
            dale_diagonals[row + col] = 0

        def add_solution():
            solution = []
            for _, col in sorted(queens):
                solution.append("." * col + "Q" + "." * (n - col - 1))
            output.append(solution)

        cols = [0] * n
        hill_diagonals = [0] * (2 * n - 1)
        dale_diagonals = [0] * (2 * n - 1)
        queens: Set[str] = set()
        output: List[List[str]] = []
        backtrack()
        return output

    def totalNQueens(self, n: int) -> int:
        # 52.N-Queens II
        visited_cols = set()
        visited_diagonals = set()
        visited_antidiagonals = set()

        def backtrack(r):
            if r == n:  # valid solution state
                return 1
            cnt = 0
            for c in range(n):
                if not (c in visited_cols or (r - c) in visited_diagonals or (r + c) in visited_antidiagonals):
                    visited_cols.add(c)
                    visited_diagonals.add(r - c)
                    visited_antidiagonals.add(r + c)
                    cnt += backtrack(r + 1)  # count the overall tally from this current state

                    visited_cols.remove(c)
                    visited_diagonals.remove(r - c)
                    visited_antidiagonals.remove(r + c)
            return cnt

        return backtrack(0)

    def maxSubArray(self, nums: List[int]) -> int:
        # 53.Maximum Subarray
        dp = nums
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
        return max(dp)

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # 54.Spiral Matrix
        result: List[int] = []
        while matrix:
            # Add the first row
            result += matrix.pop(0)
            # Add the rightmost element of each remaining row and remove it
            for row in matrix:
                if row:
                    result.append(row.pop())
            # Add the last row in reverse order and remove it
            if matrix:
                result += matrix.pop()[::-1]
            # Add the leftmost element of each remaining row and remove it
            for row in matrix[::-1]:
                if row:
                    result.append(row.pop(0))
        return result

    def canJump(self, nums: List[int]) -> bool:
        # 55.Jump Game
        if len(nums) == 1:
            return True
        can_touch = nums[0]
        index = 0
        while can_touch > index:
            if can_touch >= len(nums) - 1:
                return True
            index += 1
            can_touch = max(can_touch, index + nums[index])
        return False

    def merge_v2(self, intervals: List[List[int]]) -> List[List[int]]:
        # 56.Merge Intervals
        nitvals = len(intervals)
        if nitvals == 1:
            return intervals
        intervals.sort()
        ans: List[List[int]] = []
        curr = intervals[0]
        for i in range(1, nitvals):
            next = intervals[i]
            if curr[1] < next[0]:
                ans.append(curr)
                curr = next
            elif next[1] >= curr[1]:
                curr = [curr[0], next[1]]
        ans.append(curr)
        return ans

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # 57.Insert Interval
        intervals.append(newInterval)
        intervals.sort()
        ans: List[List[int]] = []
        curr = intervals[0]
        for i in range(1, len(intervals)):
            next = intervals[i]
            if curr[1] < next[0]:
                ans.append(curr)
                curr = next
            elif next[1] >= curr[1]:
                curr = [curr[0], next[1]]
        ans.append(curr)
        return ans

    def lengthOfLastWord(self, s: str) -> int:
        # 58.Length of Last Word
        return len(s.split()[-1])

    def generateMatrix(self, n: int) -> List[List[int]]:
        # 59.Spiral Matrix II
        matrix = [[0] * n for _ in range(n)]
        num = 1
        for layer in range((n + 1) // 2):
            # Traverse right
            for col in range(layer, n - layer):
                matrix[layer][col] = num
                num += 1
            # Traverse down
            for row in range(layer + 1, n - layer):
                matrix[row][n - layer - 1] = num
                num += 1
            # Traverse left
            for col in range(layer + 1, n - layer):
                matrix[n - layer - 1][n - col - 1] = num
                num += 1
            # Traverse up
            for row in range(layer + 1, n - layer - 1):
                matrix[n - row - 1][layer] = num
                num += 1
        return matrix

    def getPermutation(self, n: int, k: int) -> str:
        # 60.Permutation Sequence
        nums = [i for i in range(1, n + 1)]
        ans = ""
        k -= 1
        while n > 0:
            n -= 1
            index, k = divmod(k, math.factorial(n))
            ans += str(nums.pop(index))
        return ans

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 61.Rotate List
        if not head or not head.next:
            return head
        # Get the length of the list
        length = 1
        curr = head
        while curr.next:
            curr = curr.next
            length += 1
        # Connect the tail to the head
        curr.next = head
        # Find the new tail
        new_tail = head
        for _ in range(length - k % length - 1):
            new_tail = new_tail.next
        # Find the new head
        new_head = new_tail.next
        # Break the cycle
        new_tail.next = None
        return new_head

    def uniquePaths(self, m: int, n: int) -> int:
        # 62.Unique Paths
        total_steps = m + n - 2
        if total_steps < 2:
            return 1
        decay = min(m - 1, n - 1)
        up, down = 1, 1
        for _ in range(min(m - 1, n - 1)):
            up *= total_steps
            down *= decay
            total_steps -= 1
            decay -= 1
        return int(up / down)

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # 63.Unique Paths II
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if obstacleGrid[0][0] == 1:
            return 0
        obstacleGrid[0][0] = 1
        for i in range(1, m):
            obstacleGrid[i][0] = 0 if obstacleGrid[i][0] == 1 or obstacleGrid[i - 1][0] == 0 else 1
        for j in range(1, n):
            obstacleGrid[0][j] = 0 if obstacleGrid[0][j] == 1 or obstacleGrid[0][j - 1] == 0 else 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    obstacleGrid[i][j] = 0
                else:
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]
        return obstacleGrid[m - 1][n - 1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        # 64.Minimum Path Sum
        m, n = len(grid), len(grid[0])
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        for j in range(1, n):
            grid[0][j] += grid[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[m - 1][n - 1]

    def isNumber(self, s: str) -> bool:
        # 65.Valid Number
        s = s.strip()
        if not s:
            return False
        dot_seen = False
        e_seen = False
        num_seen = False
        num_after_e = False

        for i, char in enumerate(s):
            if char == "+" or char == "-":
                if i > 0 and s[i - 1] != "e" and s[i - 1] != "E":
                    return False
            elif char.isdigit():
                num_seen = True
                num_after_e = True
            elif char == ".":
                if dot_seen or e_seen:
                    return False
                dot_seen = True
            elif char == "e" or char == "E":
                if e_seen or not num_seen:
                    return False
                e_seen = True
                num_after_e = False
                if i == len(s) - 1:
                    return False
            else:
                return False
        return num_seen and num_after_e

    def plusOne(self, digits: List[int]) -> List[int]:
        # 66.Plus One
        # === Solution 1 ===
        # digit = 0
        # maxpow = len(digits) - 1
        # for i, val in enumerate(digits):
        #     digit += val * 10 ** (maxpow - i)
        # digit += 1
        # result = []
        # for i in range(len(str(digit))):
        #     result.append(int(str(digit)[i]))
        # return result
        # ==================
        result = []
        next1 = 1
        for num in digits[-1::-1]:
            result.append((num + next1) % 10)
            if (num + next1) > 9:
                next1 = 1
            else:
                next1 = 0
        if next1:
            result.append(next1)
        return result[-1::-1]

    def addBinary(self, a: str, b: str) -> str:
        # 67.Add Binary
        def num2bstr(num):
            bstr = []
            while num:
                bstr.append(str(num % 2))
                num = num // 2
            if bstr == []:
                return "0"
            return "".join(bstr[-1::-1])

        def bstr2num(bstr):
            num = 0
            maxpow = len(bstr) - 1
            for i in range(len(bstr)):
                num += int(bstr[i]) * 2 ** (maxpow - i)
            return num

        return num2bstr(bstr2num(a) + bstr2num(b))

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # 68.Text Justification
        result: List[str] = []
        line: List[str] = []
        line_len = 0
        for word in words:
            if line_len + len(word) + len(line) > maxWidth:
                for i in range(maxWidth - line_len):
                    line[i % (len(line) - 1 or 1)] += " "
                result.append("".join(line))
                line = []
                line_len = 0
            line.append(word)
            line_len += len(word)
        result.append(" ".join(line).ljust(maxWidth))
        return result

    def mySqrt(self, x: int) -> int:
        # 69.Sqrt(x)
        if x == 0 or x == 1:
            return x
        left, right = 1, x
        while left <= right:
            mid = (left + right) // 2
            squared = mid * mid
            if squared == x:
                return mid
            elif squared < x:
                left = mid + 1
            else:
                right = mid - 1
        return right

    def climbStairs(self, n: int) -> int:
        # 70.Climbing Stairs
        ways, one, two = 0, 1, 0
        for _ in range(n):
            ways = one + two
            two = one
            one = ways
        return ways

    def simplifyPath(self, path: str) -> str:
        # 71.Simplify Path
        stack: List[str] = []
        for p in path.split("/"):
            if p == "..":
                if stack:
                    stack.pop()
            elif p and p != ".":
                stack.append(p)
        return "/" + "/".join(stack)

    def minDistance(self, word1: str, word2: str) -> int:
        # 72.Edit Distance
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[m][n]

    def setZeroes(self, matrix: List[List[int]]) -> None:  # pragma: no cover
        # 73.Set Matrix Zeroes
        is_col = False
        R = len(matrix)
        C = len(matrix[0])
        for i in range(R):
            # Since first cell for both first row and first column is the same i.e. matrix[0][0]
            # We can use an additional variable for either the first row/column.
            # For this solution we are using an additional variable for the first column
            # and using matrix[0][0] for the first row.
            if matrix[i][0] == 0:
                is_col = True
            for j in range(1, C):
                # If an element is zero, we set the first element of the corresponding row and column to 0
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0

        # Iterate over the array once again and using the first row and first column, update the elements.
        for i in range(1, R):
            for j in range(1, C):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j] = 0

        # See if the first row needs to be set to zero as well
        if matrix[0][0] == 0:
            for j in range(C):
                matrix[0][j] = 0

        # See if the first column needs to be set to zero as well
        if is_col:
            for i in range(R):
                matrix[i][0] = 0

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 74.Search a 2D Matrix
        if not matrix or not matrix[0]:
            return False
        m, n = len(matrix), len(matrix[0])
        low, high = 0, m * n - 1
        while low <= high:
            mid = (low + high) // 2
            mid_element = matrix[mid // n][mid % n]
            if mid_element == target:
                return True
            elif mid_element < target:
                low = mid + 1
            else:
                high = mid - 1
        return False

    def sortColors(self, nums: List[int]) -> None:  # pragma: no cover
        # 75.Sort Colors
        ct_nums = Counter(nums)
        ans: List[int] = []
        for color in [0, 1, 2]:
            if color in ct_nums.keys():
                ans.extend([color] * ct_nums[color])
        nums[:] = ans[:]

    def minWindow(self, s: str, t: str) -> str:
        # 76.Minimum Window Substring
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if not t or not s:
            return ""
        dict_t = Counter(t)
        required = len(dict_t)
        # Filter all the characters from s into a new list along with their index.
        # The filtering criteria is that the character should be present in t.
        filtered_s = []
        for i, char in enumerate(s):
            if char in dict_t:
                filtered_s.append((i, char))
        left, right = 0, 0
        formed = 0
        window_counts: Dict[str, Union[Any, int]] = {}
        ans: Tuple[int, int, int] = int(1e5), 0, 0
        # Look for the characters only in the filtered list instead of entire s. This helps to reduce our search.
        # Hence, we follow the sliding window approach on as small list.
        while right < len(filtered_s):
            character = filtered_s[right][1]
            window_counts[character] = window_counts.get(character, 0) + 1
            if window_counts[character] == dict_t[character]:
                formed += 1
            # If the current window has all the characters in desired frequencies i.e. t is present in the window
            while left <= right and formed == required:
                character = filtered_s[left][1]
                # Save the smallest window until now.
                end = filtered_s[right][0]
                start = filtered_s[left][0]
                if end - start + 1 < ans[0]:
                    ans = (end - start + 1, start, end)
                window_counts[character] -= 1
                if window_counts[character] < dict_t[character]:
                    formed -= 1
                left += 1
            right += 1
        return "" if ans[0] == int(1e5) else s[ans[1] : ans[2] + 1]

    def combine(self, n: int, k: int) -> List[List[int]]:
        # 77.Combinations
        def backtrack(start, path):
            if len(path) == k:
                result.append(path[:])
                return
            for i in range(start, n + 1):
                path.append(i)
                backtrack(i + 1, path)
                path.pop()

        result: List[List[int]] = []
        backtrack(1, [])
        return result

    def combine_v2(self, n: int, k: int) -> List[List[int]]:
        # 77.Combinations
        def generate_combinations(elems, num):
            elems_tuple = tuple(elems)
            total = len(elems_tuple)
            curr_indices = list(range(num))
            while True:
                yield tuple(elems_tuple[i] for i in curr_indices)
                for idx in reversed(range(num)):
                    if curr_indices[idx] != idx + total - num:
                        break
                else:
                    return
                curr_indices[idx] += 1
                for j in range(idx + 1, num):
                    curr_indices[j] = curr_indices[j - 1] + 1

        return [list(combination) for combination in generate_combinations(range(1, n + 1), k)]

    def subsets(self, nums: List[int]) -> Any:  # pragma: no cover
        # 78.Subsets
        return list(chain.from_iterable(list(combinations(nums, k) for k in range(len(nums) + 1))))

    def exist(self, board: List[List[str]], word: str) -> bool:
        # 79.Word Search
        def dfs(i, j, k):
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            tmp, board[i][j] = board[i][j], "/"
            res = dfs(i - 1, j, k + 1) or dfs(i + 1, j, k + 1) or dfs(i, j - 1, k + 1) or dfs(i, j + 1, k + 1)
            board[i][j] = tmp
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False

    def deleteDuplicates_v2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 82.Remove Duplicates from Sorted List II
        # if head is None:
        #     return None
        # elif head.next is None:
        #     return head
        # duplicates: List[int] = []
        # ans = ListNode(0)
        # prev = ans
        # curr = head
        # while curr:
        #     fast = curr.next
        #     if fast is None:
        #         if curr.val not in duplicates:
        #             prev.next = ListNode(curr.val)
        #             prev = prev.next
        #         curr = fast
        #     elif curr.val != fast.val:
        #         if curr.val not in duplicates:
        #             prev.next = ListNode(curr.val)
        #             prev = prev.next
        #         curr = fast
        #     else:
        #         if curr.val not in duplicates:
        #             duplicates.append(curr.val)
        #         curr = fast.next
        # return ans.next
        # ==========================================
        if head is None:
            return None
        ans = ListNode(0)
        ans_curr = ans
        prev, curr = ListNode(-200, next=head), head
        while curr and curr.next:
            if prev.val < curr.val < curr.next.val:
                ans_curr.next = ListNode(curr.val)
                ans_curr = ans_curr.next
            prev, curr = curr, curr.next
        if prev.val < curr.val:
            ans_curr.next = ListNode(curr.val)
        return ans.next

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 83.Remove Duplicates from Sorted List
        if not head:
            return None
        curr, res = head, head
        while curr and curr.next:
            if curr.next.val != res.val:
                res.next = curr.next
                res = res.next
            curr = curr.next
        res.next = None
        return head

    def largestRectangleArea(self, heights: List[int]) -> int:
        # 84.Largest Ractangle in Histogram
        st: List[Tuple[int, int]] = []
        ans = 0
        for bar in heights + [-1]:  # add -1 to have an additional iteration
            step = 0
            while st and st[-1][1] >= bar:
                w, h = st.pop()
                step += w
                ans = max(ans, step * h)
            st.append((step + 1, bar))
        return ans

    def isScramble(self, s1: str, s2: str) -> bool:
        # 87.Scramble String
        # n = len(s1)
        # if n == 1 and s1 == s2:
        #     return True
        # for i in range(n - 1):
        #     if Counter(s1[: i + 1]) == Counter(s2[-i - 1 :]):
        #         return self.isScramble(s1[: i + 1], s2[-i - 1 :]) and self.isScramble(s1[i + 1 :], s2[: -i - 1])
        #     elif Counter(s1[: i + 1]) == Counter(s2[: i + 1]):
        #         return self.isScramble(s1[: i + 1], s2[: i + 1]) and self.isScramble(s1[i + 1 :], s2[i + 1 :])
        # return False
        # ==========================================
        memo: Dict[tuple, bool] = {}

        def _isScramble(s1: str, s2: str, memo: dict) -> bool:
            if (s1, s2) in memo:
                return memo[(s1, s2)]
            if s1 == s2:
                memo[(s1, s2)] = True
                return True
            if sorted(s1) != sorted(s2):
                memo[(s1, s2)] = False
                return False
            for i in range(1, len(s1)):
                if (_isScramble(s1[:i], s2[:i], memo) and _isScramble(s1[i:], s2[i:], memo)) or (
                    _isScramble(s1[:i], s2[-i:], memo) and _isScramble(s1[i:], s2[:-i], memo)
                ):
                    memo[(s1, s2)] = True
                    return True
            memo[(s1, s2)] = False
            return False

        return _isScramble(s1, s2, memo)

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """88.Merge Sorted Array
        Do not return anything, modify nums1 in-place instead.
        """
        a, b, write_index = m - 1, n - 1, m + n - 1

        while b >= 0:
            if a >= 0 and nums1[a] > nums2[b]:
                nums1[write_index] = nums1[a]
                a -= 1
            else:
                nums1[write_index] = nums2[b]
                b -= 1

            write_index -= 1

    def grayCode(self, n: int) -> List[int]:
        # 89.Gray Code
        ans = [0]
        for i in range(n):
            ans += [x + 2**i for x in reversed(ans)]
        return ans

    def subsetsWithDup(self, nums: List[int]) -> Any:  # pragma: no cover
        # 90.Subsets II
        nums.sort()
        subsets = chain.from_iterable(combinations(nums, k) for k in range(len(nums) + 1))
        return set(subsets)

    def numDecodings(self, s: str) -> int:
        # 91.Decode Ways
        if s[0] == "0":
            return 0
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            # If the current digit is not 0, it can be decoded as a single digit
            if s[i - 1] != "0":
                dp[i] += dp[i - 1]
            # If the last two digits form a number between 10 and 26, they can be decoded as a double-digit
            two_digit = int(s[i - 2 : i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]
        return dp[n]

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 94.Binary Tree Inorder Traversal
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right) if root else []
        # def _treenode_traver(root: Optional[TreeNode], ans: List[int]):
        #     if root is not None:
        #         _treenode_traver(root.left, ans)
        #         ans.append(root.val)
        #         _treenode_traver(root.right, ans)
        #     else:
        #         return
        #
        # self.ans = []
        # _treenode_traver(root, self.ans)
        # return self.ans

    def numTrees(self, n: int) -> int:
        # 96.Unique Binary Search Trees
        # Initialize the dp list with 1
        dp = [1] * (n + 1)
        # Calculate number of unique BSTs for each i from 2 to n
        for i in range(2, n + 1):
            total = 0
            # Calculate number of unique BSTs for each j from 1 to i
            for j in range(1, i + 1):
                total += dp[j - 1] * dp[i - j]
            dp[i] = total
        return dp[n]

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 97.Interleaving String
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0 and s1[i - 1] == s3[i + j - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
                if j > 0 and s2[j - 1] == s3[i + j - 1]:
                    dp[i][j] = dp[i][j] or dp[i][j - 1]
        return dp[m][n]

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # 98.Validate Binary Search Tree
        # ==============================================
        # S1: Two step: Traversal and IsSorted
        # ==============================================
        # def in_order_traversal(node: Optional[TreeNode]):
        #     if not node:
        #         return []
        #     return in_order_traversal(node.left) + [node.val] + in_order_traversal(node.right)
        #
        # def is_sorted(lst: list) -> bool:
        #     for i in range(len(lst) - 1):
        #         if lst[i] >= lst[i + 1]:
        #             return False
        #     return True
        #
        # values = in_order_traversal(root)
        # return is_sorted(values)
        # ==============================================
        # S2: Perform the in-order traversal while checking the
        #     order of the node values at the same time
        # ==============================================
        if not root:
            return True
        stack: list = []
        node = root
        prev_value = -math.inf
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            if prev_value >= node.val:
                return False
            prev_value = node.val
            node = node.right
        return True

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # 100.Same Tree
        if p is None and q is None:
            return True
        elif not p or not q:
            return False
        elif (p.val != q.val) or (p.left and not q.left) or (p.right and not q.right):
            return False
        else:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSameTree_v2(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # 100.Same Tree
        def check(p, q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            return True

        deq = deque(
            [
                (p, q),
            ]
        )
        while deq:
            p, q = deq.popleft()
            if not check(p, q):
                return False
            if p and q:
                deq.append((p.left, q.left))
                deq.append((p.right, q.right))
        return True

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # 101.Symmetric Tree
        def _eq_treenode(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
            if p is None and q is None:
                return True
            elif not p or not q:
                return False
            elif (p.val != q.val) or (p.left and not q.right) or (p.right and not q.left):
                return False
            else:
                return _eq_treenode(p.left, q.right) and _eq_treenode(p.right, q.left)

        return _eq_treenode(root, root)

    def isSymmetric_v2(self, root: Optional[TreeNode]) -> bool:
        # 101.Symmetric Tree
        if not root:
            return True
        deq = deque(
            [
                (root.left, root.right),
            ]
        )
        while deq:
            p, q = deq.popleft()
            if not p and not q:
                continue
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            deq.append((p.left, q.right))
            deq.append((p.right, q.left))
        return True

    def isSymmetric_v3(self, root: Optional[TreeNode]) -> bool:
        # 101.Symmetric Tree
        if not root:
            return True

        def check(left, right):
            if not left and not right:
                return True
            if left and right and left.val == right.val:
                return check(left.left, right.right) and check(left.right, right.left)
            return False

        return check(root.left, root.right)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # 102.Binary Tree Level Order Traversal
        if root is None:
            return []
        tree_levels: list = []

        def tree_level(node, level):
            if node is None:
                return
            if level > len(tree_levels):
                tree_levels.append([node.val])
            else:
                tree_levels[level - 1].append(node.val)
            tree_level(node.left, level + 1)
            tree_level(node.right, level + 1)

        tree_level(root, 1)
        return tree_levels

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # 103.Binary Tree Zigzag Level Order Traversal
        if root is None:
            return []
        tree_levels: List[List[int]] = []

        def tree_level(node, level):
            if node is None:
                return
            if level > len(tree_levels):
                tree_levels.append([node.val])
            else:
                if level % 2 == 0:
                    tree_levels[level - 1].insert(0, node.val)
                else:
                    tree_levels[level - 1].append(node.val)
            tree_level(node.left, level + 1)
            tree_level(node.right, level + 1)

        tree_level(root, 1)
        return tree_levels

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 104.Maximum Depth of Binary Tree
        if root is None:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1

    def buildTree_v2(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 105.Construct Binary Tree from Preorder and Inorder Traversal
        def build_tree(stop):
            if inorder and inorder[-1] != stop:
                root = TreeNode(preorder.pop())
                root.left = build_tree(root.val)
                inorder.pop()
                root.right = build_tree(stop)
                return root

        preorder.reverse()
        inorder.reverse()
        return build_tree(None)

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        # 106.Construct Binary Tree from Inorder and Postorder Traversal
        def build_tree(stop):
            if inorder and inorder[-1] != stop:
                root = TreeNode(postorder.pop())
                root.right = build_tree(root.val)
                inorder.pop()
                root.left = build_tree(stop)
                return root

        return build_tree(None)

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        # 108.Convert Sorted Array to Binary Search Tree
        if len(nums) == 0:
            return None
        else:
            indmid = len(nums) // 2
            return TreeNode(
                val=nums[indmid],
                left=self.sortedArrayToBST(nums[:indmid]),
                right=self.sortedArrayToBST(nums[indmid + 1 :]),
            )

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        # 109.Convert Sorted List to Binary Search Tree
        # Find the length of the linked list
        # def find_length(node: Optional[ListNode]) -> int:
        #     length = 0
        #     while node:
        #         node = node.next
        #         length += 1
        #     return length

        # # Convert the linked list to a BST using a helper function
        # def convert_to_bst(left: int, right: int) -> Optional[TreeNode]:
        #     nonlocal head
        #     if left > right:
        #         return None
        #     mid = (left + right) // 2
        #     # Recursively build the left subtree
        #     left_child = convert_to_bst(left, mid - 1)
        #     # Create the current node with the value from the linked list
        #     current = TreeNode(head.val)
        #     current.left = left_child
        #     # Move to the next value in the linked list
        #     head = head.next
        #     # Recursively build the right subtree
        #     current.right = convert_to_bst(mid + 1, right)
        #     return current

        # length = find_length(head)
        # return convert_to_bst(0, length - 1)
        def constructBST(leftHead: Optional[ListNode], rightHead: Optional[ListNode]) -> Optional[TreeNode]:
            if leftHead == rightHead or leftHead is None:
                return None
            slow, fast = leftHead, leftHead
            while fast != rightHead and fast.next != rightHead:
                slow = slow.next
                fast = fast.next.next
            root = TreeNode(slow.val)
            root.left = constructBST(leftHead, slow)
            root.right = constructBST(slow.next, rightHead)
            return root

        if not head:
            return None
        return constructBST(head, None)

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # 110.Balanced Binary Tree
        def height(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left_height = height(node.left)
            right_height = height(node.right)
            if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
                return -1
            return max(left_height, right_height) + 1

        return height(root) != -1

    def minDepth(self, root: Optional[TreeNode]) -> int:
        # 111.Minimum Depth of Binary Tree
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        left_depth = math.inf if not root.left else self.minDepth(root.left)
        right_depth = math.inf if not root.right else self.minDepth(root.right)
        return int(min(left_depth, right_depth)) + 1

    def minDepth_v2(self, root: Optional[TreeNode]) -> int:
        # 111.Minimum Depth of Binary Tree
        if not root:
            return 0
        queue: deque[TreeNode] = deque()
        queue.append(root)
        depth = 1
        while queue:  # pragma: no cover
            for _ in range(len(queue)):
                node = queue.popleft()
                if not node.left and not node.right:
                    return depth
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            depth += 1

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        # 112.Path Sum
        if not root:
            return False
        if not root.left and not root.right:
            return targetSum == root.val
        remaining_sum = targetSum - root.val
        return self.hasPathSum(root.left, remaining_sum) or self.hasPathSum(root.right, remaining_sum)

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        # 113.Path Sum II
        def dfs(node: Optional[TreeNode], remaining_sum: int, path: List[int]):
            if not node:
                return
            if not node.left and not node.right and remaining_sum == node.val:
                paths.append(path + [node.val])
            dfs(node.left, remaining_sum - node.val, path + [node.val])
            dfs(node.right, remaining_sum - node.val, path + [node.val])

        paths: List[List[int]] = []
        dfs(root, targetSum, [])
        return paths

    def flatten(self, root: Optional[TreeNode]) -> None:  # pragma: no cover
        # 114.Flatten Binary Tree to Linked List
        curr: Optional[TreeNode] = None

        def dfs(node):
            nonlocal curr
            if not node:
                return
            left, right = node.left, node.right
            node.left = None
            if curr:
                curr.right = node
                curr = curr.right
            else:
                curr = node
            dfs(left)
            dfs(right)

        dfs(root)

    def connect(self, root: Optional[Node]) -> Optional[Node]:  # pragma: no cover
        # 116.Populating Next Right Pointers in Each Node
        if not root:
            return None
        q: deque = deque([root])
        while q:
            right_node = None
            for _ in range(len(q)):
                curr = q.popleft()
                curr.next, right_node = right_node, curr
                if curr.right:
                    q.extend([curr.right, curr.left])
        return root

    def connect_v2(self, root: "Node") -> "Node":  # pragma: no cover
        # 117.Populating Next Right Pointers in Each Node II
        # if not root:
        #     return None
        q: deque = deque()
        q.append(root)
        dummy = Node(-999)  # to initialize with a not null prev
        while q:
            length = len(q)  # find level length

            prev = dummy
            for _ in range(length):  # iterate through all nodes in the same level
                popped = q.popleft()
                if popped.left:
                    q.append(popped.left)
                    prev.next = popped.left
                    prev = prev.next
                if popped.right:
                    q.append(popped.right)
                    prev.next = popped.right
                    prev = prev.next

        return root

    def generate(self, numRows: int) -> List[List[int]]:
        # 118.Pascal's Triangle
        if numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]
        else:
            pascal_tri = [[1], [1, 1]]
            for row in range(3, numRows + 1):
                curr = [1]
                for index in range(row - 2):
                    curr.append(pascal_tri[-1][index] + pascal_tri[-1][index + 1])
                curr.append(1)
                pascal_tri.append(curr)
            return pascal_tri

    def getRow(self, rowIndex: int) -> List[int]:
        # 119.Pascal's Triangle II
        ptri: List[List[int]] = [[1] * (row + 1) for row in range(rowIndex + 1)]
        if rowIndex < 2:
            return ptri[rowIndex]
        for row in range(2, rowIndex + 1):
            for index in range(row - 1):
                ptri[row][index + 1] = ptri[row - 1][index] + ptri[row - 1][index + 1]
        return ptri[rowIndex]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 120.Triangle
        if not triangle:
            return 0
        # Iterate through the triangle from the second to last row to the top
        for row in range(len(triangle) - 2, -1, -1):
            for col in range(len(triangle[row])):
                # Add the minimum of the adjacent elements in the row below
                triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1])
        return triangle[0][0]

    def maxProfit(self, prices: List[int]) -> int:
        # 121.Best Time to Buy and Sell Stock
        left, right, bestsell = 0, 1, 0
        while right < len(prices):
            sell = prices[right] - prices[left]
            if sell > 0:
                bestsell = max(sell, bestsell)
            else:
                left = right
            right += 1
        return bestsell

    def maxProfit_v2(self, prices: List[int]) -> int:
        # 122.Best Time to Buy and Sell Stock II
        ans = 0
        for i in range(len(prices) - 1):
            if prices[i] < prices[i + 1]:
                ans += prices[i + 1] - prices[i]
        return ans

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # 124.Binary Tree Maximum Path Sum
        max_sum = -1000

        def helper(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node:
                return 0
            left_gain = max(helper(node.left), 0)
            right_gain = max(helper(node.right), 0)
            max_sum = max(max_sum, node.val + left_gain + right_gain)
            return node.val + max(left_gain, right_gain)

        helper(root)
        return max_sum

    def longestConsecutive(self, nums: List[int]) -> int:
        # 128.Longest Consecutive Sequence
        if not nums:
            return 0
        nums_set = set(nums)
        longest = 0
        for num in nums_set:
            if num - 1 not in nums_set:
                current = num
                current_length = 1
                while current + 1 in nums_set:
                    current += 1
                    current_length += 1
                longest = max(longest, current_length)
        return longest

    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        # 129.Sum Root to Leaf Numbers
        def helper(node: Optional[TreeNode], current_sum: int) -> int:
            if not node:
                return 0
            current_sum = current_sum * 10 + node.val
            if not node.left and not node.right:
                return current_sum
            return helper(node.left, current_sum) + helper(node.right, current_sum)

        return helper(root, 0)

    def solve(self, board: List[List[str]]) -> None:  # pragma: no cover
        # 130.Surrounded Regions
        m, n = len(board), len(board[0])
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        seen = set()

        # checks if (row, col) is in bounds
        def valid(row, col):
            return 0 <= row < m and 0 <= col < n and board[row][col] == "O"

        # adds (row, col) and all of its valid neighbors to set
        def dfs(row, col):
            seen.add((row, col))
            for dx, dy in directions:
                new_row, new_col = row + dx, col + dy
                if valid(new_row, new_col) and (new_row, new_col) not in seen:
                    dfs(new_row, new_col)

        # process all border 'O' cells and their neighbors
        for row in range(m):
            if board[row][0] == "O":
                dfs(row, 0)
            if board[row][-1] == "O":
                dfs(row, n - 1)
        for col in range(n):
            if board[0][col] == "O":
                dfs(0, col)
            if board[-1][col] == "O":
                dfs(m - 1, col)
        # flip all unconnected 'O' cells to 'X'
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if board[i][j] == "O" and (i, j) not in seen:
                    board[i][j] = "X"
        return

    def partition(self, s: str) -> List[List[str]]:
        # 131.Palindrome Partitioning
        def is_palindrome(s: str) -> bool:
            return s == s[::-1]

        def backtrack(start: int, end: int, path: List[str]) -> None:
            if start == end:
                ans.append(path[:])
                return
            for i in range(start, end):
                if is_palindrome(s[start : i + 1]):
                    path.append(s[start : i + 1])
                    backtrack(i + 1, end, path)
                    path.pop()

        ans: List[List[str]] = []
        backtrack(0, len(s), [])
        return ans

    def cloneGraph(self, node: "Node") -> "Node":  # pragma: no cover
        # 133.Clone Graph
        def dfs(node):
            if node in visited:
                return visited[node]
            cloned_node = Node(node.val)
            visited[node] = cloned_node
            for neighbor in node.neighbors:
                cloned_node.neighbors.append(dfs(neighbor))
            return cloned_node

        visited: Dict["Node", "Node"] = {}
        return dfs(node)

    def singleNumber(self, nums: List[int]) -> int:
        # 136.Single Number
        # count = Counter(nums)
        # for key in count.keys():
        #     if count[key] == 1:
        #         return int(key)
        # return -1
        # ========================
        count = Counter(nums)
        return count.most_common()[-1][0]

    def singleNumber_v2(self, nums: List[int]) -> int:
        # 137.Single Number II
        count: defaultdict = defaultdict(int)
        for num in nums:
            count[num] += 1
        for key in count.keys():
            if count[key] == 1:
                return int(key)
        return -1

    def copyRandomList(self, head: "Optional[NodeTwo]") -> "Optional[NodeTwo]":  # pragma: no cover
        # 138.Copy List with Random Pointer
        if not head:
            return None
        # create a new linked list with the same value
        # and insert it between the original node and the next node
        node = head
        while node:
            new_node = NodeTwo(node.val)
            new_node.next = node.next
            node.next = new_node
            node = new_node.next
        # copy the random pointer for each new node
        node = head
        while node:
            if node.random:
                node.next.random = node.random.next
            node = node.next.next
        # extract the new list
        new_head = head.next
        node = head
        while node.next:
            temp = node.next
            node.next = temp.next
            node = temp
        return new_head

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # 139.Word Break
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        return dp[-1]

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # 141.Linked List Cycle
        if head is None:
            return False
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 142.Linked List Cycle II
        if head is None:
            return None
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

    def reorderList(self, head: Optional[ListNode]) -> None:  # pragma: no cover
        # 143.Reorder List
        if not head:
            return
        # Find the middle of linked list [Problem 876]
        # in 1->2->3->4->5->6 find 4
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # Reverse the second half [Problem 206]
        # convert 1->2->3->4->5->6 into 1->2->3->4 and 6->5->4
        # reverse the second half in-place
        prev, curr = None, slow
        while curr and curr.next:
            curr.next, prev, curr = prev, curr, curr.next
        # Merge two sorted linked lists [Problem 21]
        # merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        first, second = head, prev
        while second and second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 144.Binary Tree Preorder Traversal
        ans = []

        def preorder_helper(node):
            if node is not None:
                ans.append(node.val)
                preorder_helper(node.left)
                preorder_helper(node.right)

        preorder_helper(root)
        return ans

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 145.Binary Tree Postorder Traversal
        ans = []

        def preorder_helper(node):
            if node is not None:
                ans.append(node.val)
                preorder_helper(node.right)
                preorder_helper(node.left)

        preorder_helper(root)
        ans.reverse()
        return ans

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 148.Sort List
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        mid, slow.next = slow.next, None
        left, right = self.sortList(head), self.sortList(mid)
        h = res = ListNode(0)
        while left and right:
            if left.val < right.val:
                h.next, left = left, left.next
            else:
                h.next, right = right, right.next
            h = h.next
        h.next = left if left else right
        return res.next

    def maxPoints(self, points: List[List[int]]) -> int:
        # 149.Max Points on a Line
        if len(points) <= 2:
            return len(points)
        max_points = 0
        for i in range(len(points)):
            same_points = 1
            slope_count: Dict[float, int] = {}
            for j in range(i + 1, len(points)):
                if points[i] == points[j]:
                    same_points += 1
                else:
                    slope = (
                        float("inf")
                        if points[i][0] == points[j][0]
                        else (points[i][1] - points[j][1]) / (points[i][0] - points[j][0])
                    )
                    slope_count[slope] = slope_count.get(slope, 0) + 1
            max_points = max(max_points, same_points + max(slope_count.values(), default=0))
        return max_points

    def evalRPN(self, tokens: List[str]) -> int:
        # 150.Evaluate Reverse Polish Notation
        operators = ("+", "-", "*", "/")
        queue: deque = deque()
        for token in tokens:
            if token not in operators:
                queue.append(token)
            else:
                num1 = int(queue.pop())
                num2 = int(queue.pop())
                if token == "+":
                    num = num2 + num1
                elif token == "-":
                    num = num2 - num1
                elif token == "*":
                    num = num2 * num1
                else:
                    num = int(num2 / num1)
                queue.append(str(num))
        return int(queue.pop())

    def reverseWords(self, s: str) -> str:
        # 151.Reverse Words in a String
        words = re.findall(r"\w+", s)
        words.reverse()
        return " ".join(words)

    def maxProduct(self, nums: List[int]) -> int:
        # 152.Maximum Product Subarray
        # Initialize variables to keep track of maximum and minimum product
        max_product = nums[0]
        min_product = nums[0]
        result = max_product
        # Iterate through the given array
        for i in range(1, len(nums)):
            # Keep track of the maximum and minimum product seen so far
            temp_max = max_product
            temp_min = min_product
            # Update the maximum and minimum product using the current element
            max_product = max(nums[i], temp_max * nums[i], temp_min * nums[i])
            min_product = min(nums[i], temp_max * nums[i], temp_min * nums[i])
            # Update the result with the maximum product seen so far
            result = max(result, max_product)
        return result

    def findMin(self, nums: List[int]) -> int:
        # 153.Find Minimum in Rotated Sorted Array
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:  # pragma: no cover
        # 160.Intersection of Two Linked Lists
        # find the length of the linked lists
        lenA, lenB = 0, 0
        currA, currB = headA, headB
        while currA:
            lenA += 1
            currA = currA.next
        while currB:
            lenB += 1
            currB = currB.next
        # set pointers to the same distance from the end of each linked list
        currA, currB = headA, headB
        if lenA > lenB:
            for _ in range(lenA - lenB):
                currA = currA.next
        else:
            for _ in range(lenB - lenA):
                currB = currB.next
        # traverse both linked lists together until a common node is found
        while currA and currB:
            if currA == currB:
                return currA
            currA = currA.next
            currB = currB.next
        # no common node found
        return None

    def findPeakElement(self, nums: List[int]) -> int:
        # 162.Find Peak Element
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left

    def twoSum(self, numbers: List[int], target: int) -> Optional[List[int]]:
        # 167.Two Sum 2 - Input Array Is Sorted
        # left, right = 0, len(numbers) - 1
        # while left < right:
        #     current_sum = numbers[left] + numbers[right]
        #     if current_sum == target:
        #         return [left + 1, right + 1]
        #     elif current_sum < target:
        #         left += 1
        #     else:
        #         right -= 1
        # return []  # This line is not necessary as the problem states there is exactly one solution.
        cont = Counter(numbers)
        for i in cont:
            if target - i in cont:
                ind1 = numbers.index(i)
                ind2 = numbers[ind1 + 1 :].index(target - i) + ind1 + 1
                return [ind1 + 1, ind2 + 1]
        return []

    def majorityElement(self, nums: List[int]) -> int:
        # 169.Majority Element
        # return Counter(nums).most_common(1)[0][0]
        # ====================================
        nums.sort()
        return nums[len(nums) // 2]

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        # 187.Repeated DNA Sequences
        n = len(s)
        ans: List[str] = []
        if n <= 10:
            return ans
        dna_sequence = Counter(s[i : i + 10] for i in range(n - 9))
        for key in dna_sequence.keys():
            if dna_sequence[key] > 1:
                ans.append(key)
        return ans

    def rotate(self, nums: List[int], k: int) -> None:  # pragma: no cover
        """189.Rotate Array
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[k:], nums[:k] = nums[:-k], nums[-k:]

    def reverseBits(self, n: int) -> int:  # pragma: no cover
        # 190.Reverse Bits
        ans = 0
        for _ in range(32):
            ans <<= 1
            ans |= n & 1
            n >>= 1
        return ans

    def hammingWeight(self, n: int) -> int:
        # 191.Number of 1 Bits
        count = 0
        while n > 0:
            if n % 2 == 1:
                count += 1
            n = n // 2
        return count

    def rob(self, nums: List[int]) -> int:
        # 198.House Robber
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        # Initialize two variables to track the maximum money robbed with and without robbing the current house
        rob_curr, dont_rob_curr = 0, 0
        for num in nums:
            rob_next = dont_rob_curr + num
            dont_rob_next = max(rob_curr, dont_rob_curr)
            rob_curr, dont_rob_curr = rob_next, dont_rob_next
        return max(rob_curr, dont_rob_curr)

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # 199.Binary Tree Right Side View
        if not root:
            return []
        ans: List[int] = []
        queue = deque([root])
        while queue:
            ans.append(queue[-1].val)
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return ans

    def numIslands(self, grid: List[List[str]]) -> int:
        # 200.Number of Islands
        m, n = len(grid), len(grid[0])

        def dfs(grid, row, col):
            if row < 0 or row >= m or col < 0 or col >= n or grid[row][col] == "0":
                return
            grid[row][col] = "0"
            dfs(grid, row - 1, col)
            dfs(grid, row + 1, col)
            dfs(grid, row, col - 1)
            dfs(grid, row, col + 1)

        num_islands = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == "1":
                    num_islands += 1
                    dfs(grid, row, col)
        return num_islands


class LRUCache:  # pragma: no cover
    # 146.LRU Cache
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.cache.keys():
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache.keys():
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LRUCache_v2:  # pragma: no cover
    # 146.LRU Cache
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.cap = capacity
        self.head = self.Node(-1, -1)
        self.tail = self.Node(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.m: Dict[int, Node] = {}

    def addNode(self, newnode):
        temp = self.head.next
        newnode.next = temp
        newnode.prev = self.head
        self.head.next = newnode
        temp.prev = newnode

    def deleteNode(self, delnode):
        prevv = delnode.prev
        nextt = delnode.next
        prevv.next = nextt
        nextt.prev = prevv

    def get(self, key: int) -> int:
        if key in self.m:
            resNode = self.m[key]
            ans = resNode.val
            del self.m[key]
            self.deleteNode(resNode)
            self.addNode(resNode)
            self.m[key] = self.head.next
            return ans
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.m:
            curr = self.m[key]
            del self.m[key]
            self.deleteNode(curr)
        if len(self.m) == self.cap:
            del self.m[self.tail.prev.key]
            self.deleteNode(self.tail.prev)
        self.addNode(self.Node(key, value))
        self.m[key] = self.head.next


class MinStack:  # pragma: no cover
    # 155.Min Stack
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self) -> None:
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


class BSTIterator:  # pragma: no cover
    # 173.Binary Search Tree Iterator
    def __init__(self, root: Optional[TreeNode]):
        self.stack: List[TreeNode] = []
        self._leftmost_inorder(root)

    def _leftmost_inorder(self, root: Optional[TreeNode]) -> None:
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        topmost_node = self.stack.pop()
        if topmost_node.right:
            self._leftmost_inorder(topmost_node.right)
        return topmost_node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
