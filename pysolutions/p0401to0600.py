from collections import Counter
from typing import List

from .utils import Node


class Pro0401To0600:
    def __init__(self):
        pass

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
        if flag:
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

    def preorder(self, root: Node) -> List[int]:
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
