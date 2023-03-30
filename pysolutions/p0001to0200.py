import math
import re
from collections import Counter, deque
from typing import List, Optional

from .utils import ListNode, Node, TreeNode


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

    def maxSubArray(self, nums: List[int]) -> int:
        # 53.Maximum Subarray
        dp = nums
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
        return max(dp)

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

    def lengthOfLastWord(self, s: str) -> int:
        # 58.Length of Last Word
        return len(s.split()[-1])

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
        memo = {}

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

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 104.Maximum Depth of Binary Tree
        if root is None:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1

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

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        # 112.Path Sum
        if not root:
            return False
        if not root.left and not root.right:
            return targetSum == root.val
        remaining_sum = targetSum - root.val
        return self.hasPathSum(root.left, remaining_sum) or self.hasPathSum(root.right, remaining_sum)

    def connect(self, root: Optional[Node]) -> Optional[Node]:  # pragma: no cover
        # 116.Populating Next Right Pointers in Each Node
        if not root:
            return None
        q = deque([root])
        while q:
            right_node = None
            for _ in range(len(q)):
                curr = q.popleft()
                curr.next, right_node = right_node, curr
                if curr.right:
                    q.extend([curr.right, curr.left])
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
