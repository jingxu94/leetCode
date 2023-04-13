from collections import Counter, deque
from typing import List, Optional, Set

from .utils import ListNode, TreeNode


class Node:  # pragma: no cover
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Pro0401To0600:
    def __init__(self):
        pass

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        # 404.Sum of Left Leaves
        left_leaf = [0]

        def helper(node, flag):
            if node.left is not None:
                helper(node.left, 1)
            if node.right is not None:
                helper(node.right, 0)
            if node.left is None and node.right is None:
                if flag:
                    left_leaf.append(node.val)
                return

        if root is None:
            return 0
        else:
            helper(root, 0)
            return sum(left_leaf)

    def longestPalindrome(self, s: str) -> int:
        # 409.Longest Palindrome
        single, double, flag = 0, 0, 0
        count = Counter(s)
        for key in count.keys():
            if count[key] % 2 == 0:
                double += count[key]
            else:
                flag = 1
                single += count[key] - 1
        if flag == 1:
            single += 1
        return single + double

    def fizzBuzz(self, n: int) -> List[str]:
        # 412.Fizz Buzz
        answer = []
        for num in range(1, n + 1):
            if num % 3 == 0 and num % 5 == 0:
                answer.append("FizzBuzz")
            elif num % 3 == 0:
                answer.append("Fizz")
            elif num % 5 == 0:
                answer.append("Buzz")
            else:
                answer.append(str(num))
        return answer

    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # 413.Arithmetic Slices
        n = len(nums)
        if n < 3:
            return 0
        dp = [0] * n
        total_arithmetic_slices = 0
        for i in range(2, n):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1
                total_arithmetic_slices += dp[i]
        return total_arithmetic_slices

    def addStrings(self, num1: str, num2: str) -> str:
        # 415.Add String
        return str(eval(num1 + "+" + num2))

    def arrangeCoins(self, n: int) -> int:
        # 441.Arranging Coins
        left, right = 0, n
        while left <= right:
            mid = (left + right) // 2
            total_coins = ((1 + mid) * mid) // 2
            if total_coins == n:
                return mid
            elif total_coins > n:
                right = mid - 1
            else:
                left = mid + 1
        return right

    def characterReplacement(self, s: str, k: int) -> int:
        # 424.Longest Repeating Character Replacement
        left, right = 0, 0
        max_count = 0
        count = [0] * 26
        while right < len(s):
            count[ord(s[right]) - ord("A")] += 1
            max_count = max(max_count, count[ord(s[right]) - ord("A")])
            if right - left + 1 - max_count > k:
                count[ord(s[left]) - ord("A")] -= 1
                left += 1
            right += 1
        return right - left

    def levelOrder(self, root: "Node") -> List[List[int]]:  # pragma: no cover
        # 429.N-ary Tree Level Order Traversal
        ans: List[List[int]] = []
        if not root:
            return ans
        level = [root]
        while level:
            curr_level: List[int] = []
            next_level: List["Node"] = []
            for node in level:
                curr_level.append(node.val)
                next_level.extend(node.children)
            ans.append(curr_level)
            level = next_level
        return ans

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # 435.Non-overlapping Intervals
        intervals.sort(key=lambda x: x[1])
        count = 0
        end = intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                count += 1
            else:
                end = intervals[i][1]
        return count

    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 438.Find All Anagrams in a String
        if len(s) < len(p):
            return []
        ct_p = Counter(p)
        ct_s = Counter(s[: len(p)])
        ans = []
        if ct_p == ct_s:
            ans.append(0)
        for i in range(len(p), len(s)):
            ct_s[s[i - len(p)]] -= 1
            if ct_s[s[i - len(p)]] == 0:
                del ct_s[s[i - len(p)]]
            ct_s[s[i]] += 1
            if ct_p == ct_s:
                ans.append(i - len(p) + 1)
        return ans

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 445.Add Two Numbers II
        def reverse(head):
            prev = None
            while head:
                next = head.next
                head.next = prev
                prev = head
                head = next
            return prev

        l1 = reverse(l1)
        l2 = reverse(l2)
        dummy = ListNode(0)
        curr = dummy
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            curr.next = ListNode(carry % 10)
            curr = curr.next
            carry //= 10
        return reverse(dummy.next)

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        # 450.Delete Node in a BST
        if not root:
            return None
        if root.val == key:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            curr = root.right
            while curr.left:
                curr = curr.left
            root.val = curr.val
            root.right = self.deleteNode(root.right, curr.val)
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
        return root

    def repeatedSubstringPattern(self, s: str) -> bool:
        # 459.Repeated Substring Pattern
        # half = len(s) // 2 + 1
        # for i in range(1, half):
        #     elements = s.split(s[:i])
        #     if all(ele == "" for ele in elements):
        #         return True
        # return False
        # ==============================
        n = len(s)
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                if s[:i] * (n // i) == s:
                    return True
        return False

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 496.Next Greater Element1
        ans = list([-1 for _ in range(len(nums1))])
        for i, num in enumerate(nums1):
            index = nums2.index(num)
            if index < len(nums2) - 1:
                for j, check in enumerate(nums2[index + 1 :]):
                    if check > num:
                        ans[i] = check
                        break
        return ans

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        # 503.Next Greater Element II
        n = len(nums)
        ans: List[int] = [-1] * n
        stack: List[int] = []
        for i in range(2 * n):
            while stack and nums[stack[-1]] < nums[i % n]:
                index = stack.pop()
                ans[index] = nums[i % n]
            if i < n:
                stack.append(i)
        return ans

    def fib(self, n: int) -> int:
        # 509.Fibonacci Number
        if n == 0 or n == 1:
            return n
        else:
            prev, ans = 0, 1
            for _ in range(n - 1):
                prev, ans = ans, ans + prev
            return ans

    def longestPalindromeSubseq(self, s: str) -> int:
        # 516.Longest Palindromic Subsequence
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][n - 1]

    def change(self, amount: int, coins: List[int]) -> int:
        # 518.Coin Change II
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        return dp[amount]

    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        # 542.0 1 Matrix
        m, n = len(mat), len(mat[0])
        dx = [0, 1, 0, -1, 0]
        queue: deque = deque([])
        for row in range(m):
            for col in range(n):
                if mat[row][col] == 0:
                    queue.append((row, col))
                else:
                    mat[row][col] = -1
        while queue:
            row, col = queue.popleft()
            for i in range(4):
                nr, nc = row + dx[i], col + dx[i + 1]
                if nr < 0 or nr == m or nc < 0 or nc == n or mat[nr][nc] != -1:
                    continue
                mat[nr][nc] = mat[row][col] + 1
                queue.append((nr, nc))
        return mat

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        # 547.Number of Provinces
        checked: Set[int] = set()

        def dfs(isConnected, checked, i):
            if i in checked:
                return 0
            checked.add(i)
            for j in range(len(isConnected[i])):
                if isConnected[i][j] == 1:
                    dfs(isConnected, checked, j)
            return 1

        provinces = 0
        for i in range(len(isConnected)):
            provinces += dfs(isConnected, checked, i)
        return provinces

    def nextGreaterElement_v2(self, n: int) -> int:
        # 556.Next Greater Element III
        digits = list(str(n))
        length = len(digits)
        i = length - 2
        while i >= 0 and digits[i] >= digits[i + 1]:
            i -= 1
        if i == -1:
            return -1
        j = length - 1
        while j > i and digits[j] <= digits[i]:
            j -= 1
        digits[i], digits[j] = digits[j], digits[i]
        digits[i + 1 :] = sorted(digits[i + 1 :])
        result = int("".join(digits))
        if result > 2**31 - 1:
            return -1
        return result

    def subarraySum(self, nums: List[int], k: int) -> int:
        # 560.Subarray Sum Equals K
        count = 0
        cumulative_sum = 0
        sum_count = {0: 1}
        for num in nums:
            cumulative_sum += num
            if cumulative_sum - k in sum_count:
                count += sum_count[cumulative_sum - k]
            if cumulative_sum in sum_count:
                sum_count[cumulative_sum] += 1
            else:
                sum_count[cumulative_sum] = 1
        return count

    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        # 566.Reshape the Matrix
        m, n = len(mat), len(mat[0])
        if m * n != r * c:
            return mat
        cnt = 0
        ans = []
        row = []
        for i in range(m):
            for j in range(n):
                row.append(mat[i][j])
                cnt += 1
                if cnt == c:
                    ans.append(row)
                    row = []
                    cnt = 0
        return ans

    def reverseWords(self, s: str) -> str:
        # 557.Reverse Words in a String 3
        words = s.split(" ")
        reversed_words = []
        for word in words:
            reversed_words.append(word[::-1])
        return " ".join(reversed_words)

    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 567.Permutation in String
        if s1 in s2:
            return True
        ct_s1, ct_s2 = Counter(s1), Counter(s2)
        if not all(ct_s1[key] <= ct_s2.get(key, 0) for key in ct_s1.keys()):
            return False
        for i in range(len(s2) - len(s1) + 1):
            if ct_s1 == Counter(s2[i : i + len(s1)]):
                return True
        return False

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # 572.Subtree of Another Tree
        def serialize(node, tree_str):
            if node is None:
                tree_str.append("#")
                return
            tree_str.append("^")
            tree_str.append(str(node.val))
            serialize(node.left, tree_str)
            serialize(node.right, tree_str)

        def kmp(needle, haystack):
            m, n = len(needle), len(haystack)
            if n < m:
                return False
            lps = [0] * m
            prev = 0
            i = 1
            while i < m:
                if needle[i] == needle[prev]:
                    prev += 1
                    lps[i] = prev
                    i += 1
                else:
                    if prev == 0:
                        lps[i] = 0
                        i += 1
                    else:
                        prev = lps[prev - 1]
            needle_pointer = haystack_pointer = 0
            while haystack_pointer < n:
                if haystack[haystack_pointer] == needle[needle_pointer]:
                    needle_pointer += 1
                    haystack_pointer += 1
                    if needle_pointer == m:
                        return True
                else:
                    if needle_pointer == 0:
                        haystack_pointer += 1
                    else:
                        needle_pointer = lps[needle_pointer - 1]
            return False

        root_list: List[str] = []
        serialize(root, root_list)
        r = "".join(root_list)
        subroot_list: List[str] = []
        serialize(subRoot, subroot_list)
        s = "".join(subroot_list)
        return kmp(s, r)

    def preorder(self, root: Node) -> List[int]:  # pragma: no cover
        # 589.N-ary Tree Preorder Traversal
        if root is None:
            return []
        stack, ans = [root], []
        while stack:
            node = stack.pop()
            ans.append(node.val)
            for child in reversed(node.children):
                stack.append(child)
        return ans
