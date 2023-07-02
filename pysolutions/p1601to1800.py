import math
from collections import Counter, deque
from typing import Dict, List, Optional
from .utils import ListNode


class Pro1601To1800:
    def __init__(self):
        pass

    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        # 1601.Maximum Number of Achievable Transfer Requests
        def solve(ind, building):
            if ind == m:
                for i in arr:
                    if i != 0:
                        return -int(1e9)
                return 0
            # Request[ind] is not taken
            not_take = solve(ind + 1, arr)
            # Request[ind] is taken
            building[requests[ind][0]] -= 1
            building[requests[ind][1]] += 1
            take = 1 + solve(ind + 1, arr)
            building[requests[ind][0]] += 1
            building[requests[ind][1]] -= 1
            return max(take, not_take)

        m = len(requests)
        arr = [0 for i in range(n + 1)]
        return solve(0, arr)

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

    def checkArithmeticSubarrays(self, nums: List[int], left: List[int], right: List[int]) -> List[bool]:
        # 1630.Arithmetic Subarrays
        ans: List[bool] = []
        for li, ri in zip(left, right):
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

    def closeStrings(self, word1: str, word2: str) -> bool:
        # 1657.Determine if Two Strings Are Close
        if len(word1) != len(word2):
            return False
        count1 = Counter(word1)
        count2 = Counter(word2)
        if count1.keys() != count2.keys():
            return False
        if sorted(count1.values()) != sorted(count2.values()):
            return False
        return True

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

    def maxOperations(self, nums: List[int], k: int) -> int:
        # 1679.Max Number of K-Sum Pairs
        nums.sort()
        i, j = 0, len(nums) - 1
        ans = 0
        while i < j:
            if nums[i] + nums[j] == k:
                ans += 1
                i += 1
                j -= 1
            elif nums[i] + nums[j] < k:
                i += 1
            else:
                j -= 1
        return ans

    def distanceLimitedPathsExist(self, n: int, edgeList: List[List[int]], queries: List[List[int]]) -> List[bool]:
        # 1697.Checking Existence of Edge Length Limited Paths
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.size = [1] * n

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if self.size[px] < self.size[py]:
                    px, py = py, px
                self.parent[py] = px
                self.size[px] += self.size[py]
                return True

        edgeList.sort(key=lambda x: x[2])
        queries_set = sorted([(i, q[0], q[1], q[2]) for i, q in enumerate(queries)], key=lambda x: x[3])
        uf = UnionFind(n)
        ans = [False] * len(queries_set)
        i = 0
        for query in queries_set:
            while i < len(edgeList) and edgeList[i][2] < query[3]:
                uf.union(edgeList[i][0], edgeList[i][1])
                i += 1
            ans[query[0]] = uf.find(query[1]) == uf.find(query[2])
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

    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 1721.Swapping Nodes in a Linked List
        dummy = ListNode(0, head)
        fast = slow = dummy
        for _ in range(k):
            fast = fast.next
        first = fast
        while fast:
            fast = fast.next
            slow = slow.next
        second = slow
        first.val, second.val = second.val, first.val
        return dummy.next

    def largestAltitude(self, gain: List[int]) -> int:
        # 1732.Find the Highest Altitude
        ans = 0
        altitude = 0
        for val in gain:
            altitude += val
            ans = max(ans, altitude)
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

    def maxScore(self, nums: List[int]) -> int:
        # 1799.Maximize Score After N Operations
        # Determine number of elements
        num_elems = len(nums)
        # Construct matrix of greatest common divisors for all pairs of elements
        gcd_pairs = [[math.gcd(nums[i], nums[j]) for j in range(num_elems)] for i in range(num_elems)]
        # Use dynamic programming to find maximum score
        max_scores = [0] * (1 << num_elems)
        for state in range(1, 1 << num_elems):
            num_selected = bin(state).count("1")
            # Skip states with odd number of selected elements
            if num_selected % 2 == 1:
                continue
            # Iterate over all pairs of selected elements
            for i in range(num_elems):
                if not (state & (1 << i)):
                    continue
                for j in range(i + 1, num_elems):
                    if not (state & (1 << j)):
                        continue
                    # Compute score for current state based on previous state and current pair of elements
                    prev_state = state ^ (1 << i) ^ (1 << j)
                    current_score = max_scores[prev_state] + num_selected // 2 * gcd_pairs[i][j]
                    # Update maximum score for current state
                    max_scores[state] = max(max_scores[state], current_score)
        # Return maximum score for state with all elements selected
        return max_scores[(1 << num_elems) - 1]


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
