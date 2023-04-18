import math
from collections import Counter, deque
from typing import Dict, List


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

    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        # 1630.Arithmetic Subarrays
        ans: List[bool] = []
        for li, ri in zip(l, r):
            flag = True
            subarray = list(nums[i] for i in range(li, ri + 1))
            if len(subarray) > 2:
                subarray.sort()
                gap = subarray[1] - subarray[0]
                for i in range(1, len(subarray) - 1):
                    if subarray[i + 1] - subarray[i] != gap:
                        flag = False
                        break
            ans.append(flag)
        return ans

    def numWays(self, words: List[str], target: str) -> int:
        # 1639.Number of Ways to Form a Target String Given a Dictionary
        n = len(words[0])
        m = len(target)
        mod = 10**9 + 7
        dp = [0] * (m + 1)
        dp[0] = 1
        count = [[0] * 26 for _ in range(n)]
        for i in range(n):
            for word in words:
                count[i][ord(word[i]) - ord("a")] += 1
        for i in range(n):
            for j in range(m - 1, -1, -1):
                dp[j + 1] = (dp[j + 1] + dp[j] * count[i][ord(target[j]) - ord("a")]) % mod
        return dp[m]

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

    def findBall(self, grid: List[List[int]]) -> List[int]:
        # 1706.Where Will the Ball Fall
        m, n = len(grid), len(grid[0])
        ans: List[int] = []
        for i in range(n):
            x, y = 0, i
            while x < m:
                if grid[x][y] == 1:
                    if y == n - 1 or grid[x][y + 1] == -1:
                        ans.append(-1)
                        break
                    else:
                        x += 1
                        y += 1
                else:
                    if y == 0 or grid[x][y - 1] == 1:
                        ans.append(-1)
                        break
                    else:
                        x += 1
                        y -= 1
            else:
                ans.append(y)
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


class ParkingSystem:  # pragma: no cover
    # 1603.Design Parking System
    def __init__(self, big: int, medium: int, small: int):
        self.parking = [big, medium, small]

    def addCar(self, carType: int) -> bool:
        if self.parking[carType - 1] > 0:
            self.parking[carType - 1] -= 1
            return True
        return False


class AuthenticationManager:  # pragma: no cover
    # 1797.Design Authentication Manager
    def __init__(self, timeToLive: int):
        self.timeToLive = timeToLive
        self.tokens: Dict[str, int] = {}

    def generate(self, tokenId: str, currentTime: int) -> None:
        self.tokens[tokenId] = currentTime + self.timeToLive

    def renew(self, tokenId: str, currentTime: int) -> None:
        if tokenId in self.tokens and self.tokens[tokenId] > currentTime:
            self.tokens[tokenId] = currentTime + self.timeToLive

    def countUnexpiredTokens(self, currentTime: int) -> int:
        return sum(1 for token in self.tokens.values() if token > currentTime)
