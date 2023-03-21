import math
from collections import Counter, deque
from typing import List


class Pro1601To1800:
    def __init__(self):
        pass

    def specialArray(self, nums: List[int]) -> int:
        # 1608.Special Array With X Elements Greater Than or Equal X
        count = Counter(nums)

        def nbigger(num):
            total = 0
            for key in count.keys():
                if key >= num:
                    total += count[key]
            return total

        left, right = 0, len(nums)
        while left <= right:
            mid = (left + right) // 2
            bigger = nbigger(mid)
            if bigger == mid:
                return mid
            elif bigger > mid:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def maximumWealth(self, accounts: List[List[int]]) -> int:
        # 1672.Richest Customer Wealth
        rich = []
        for customer in accounts:
            rich.append(sum(customer))
        return max(rich)

    def interpret(self, command: str) -> str:
        # 1678.Goal Parser Interpretation
        queue = deque(command)
        ans = ""
        while queue:
            fst = queue.popleft()
            if fst == "G":
                ans += fst
            else:
                sec = queue.popleft()
                if sec == ")":
                    ans += "o"
                else:
                    ans += "al"
                    queue.popleft()
                    queue.popleft()
        return ans

    def mergeAlternately(self, word1: str, word2: str) -> str:
        # 1768.Merge Strings Alternately
        qa, qb = deque(word1), deque(word2)
        ans = ""
        while qa and qb:
            ans += qa.popleft() + qb.popleft()
        if qa:
            while qa:
                ans += qa.popleft()
        if qb:
            while qb:
                ans += qb.popleft()
        return ans

    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        # 1779.Find Nearest Point That Has the Same X or Y Coordinate
        index, smd = 0, math.inf
        for i, point in enumerate(points):
            if x == point[0] or y == point[1]:
                manhattan_distance = abs(x - point[0]) + abs(y - point[1])
                if manhattan_distance < smd:
                    smd = manhattan_distance
                    index = i
        if smd == math.inf:
            return -1
        return index

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        # 1790.Check if One String Swap Can Make Strings Equal
        ndiff = 0
        diff_ind = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                ndiff += 1
                diff_ind.append(i)
            if ndiff == 2:
                if s1[diff_ind[0]] != s2[diff_ind[1]] or s1[diff_ind[1]] != s2[diff_ind[0]]:
                    return False
            if ndiff > 2:
                return False
        if ndiff == 1:
            return False
        return True
