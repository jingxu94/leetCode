from collections import Counter, deque
from typing import List, Optional

from .utils import TreeNode


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

    def fib(self, n: int) -> int:
        # 509.Fibonacci Number
        if n == 0 or n == 1:
            return n
        else:
            prev, ans = 0, 1
            for _ in range(n - 1):
                prev, ans = ans, ans + prev
            return ans

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
