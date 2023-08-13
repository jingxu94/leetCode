import heapq
import string
from collections import defaultdict
from typing import List


class Pro2201To2400:
    def __init__(self) -> None:
        pass

    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        # 2215.Find the Difference of Two Arrays
        return [list(set(nums1) - set(nums2)), list(set(nums2) - set(nums1))]

    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        # 2218.Maximum Value of K Coins From Piles
        dp = [[0] * (k + 1) for _ in range(len(piles) + 1)]
        for i in range(1, len(piles) + 1):
            for j in range(1, k + 1):
                curr = 0
                for x in range(min(len(piles[i - 1]), j)):
                    curr += piles[i - 1][x]
                    dp[i][j] = max(dp[i][j], curr + dp[i - 1][j - x - 1])
                dp[i][j] = max(dp[i][j], dp[i - 1][j])
        return dp[len(piles)][k]

    def largestVariance(self, s: str) -> int:
        # 2272.Substring With Largest Variance
        d: defaultdict = defaultdict(list)
        for i, c in enumerate(s):  # for each letter, create a list of its indices
            d[c].append(i)
        ans = 0
        for x, chr1 in enumerate(string.ascii_lowercase):  # character 1
            for chr2 in string.ascii_lowercase[x + 1 :]:  # character 2
                if chr1 == chr2 or chr1 not in d or chr2 not in d:
                    continue
                prefix = i = p1 = p2 = 0
                hi = hi_idx = lo = lo_idx = 0
                n1, n2 = len(d[chr1]), len(d[chr2])
                while p1 < n1 or p2 < n2:  # two pointers
                    if p1 < n1 and p2 < n2:
                        if d[chr1][p1] < d[chr2][p2]:
                            prefix, p1 = prefix + 1, p1 + 1  # count prefix
                        else:
                            prefix, p2 = prefix - 1, p2 + 1
                    elif p1 < n1:
                        prefix, p1 = prefix + 1, p1 + 1
                    else:
                        prefix, p2 = prefix - 1, p2 + 1
                    if prefix > hi:  # update high value
                        hi, hi_idx = prefix, i
                    if prefix < lo:  # update low value
                        lo, lo_idx = prefix, i
                    ans = max(ans, min(prefix - lo, i - lo_idx - 1))
                    # update ans by calculate difference, i-lo_idx-1 is to handle
                    # when only one elements are showing up
                    ans = max(ans, min(hi - prefix, i - hi_idx - 1))
                    i += 1
        return ans

    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        # 2300.Successful Pairs of Spells and Potions
        sorted_spells = [(spell, index) for index, spell in enumerate(spells)]
        # Sort the 'spells with index' and 'potions' array in increasing order.
        sorted_spells.sort()
        potions.sort()
        answer = [0] * len(spells)
        m = len(potions)
        potion_index = m - 1
        # For each 'spell' find the respective 'minPotion' index.
        for spell, index in sorted_spells:
            while potion_index >= 0 and (spell * potions[potion_index]) >= success:
                potion_index -= 1
            answer[index] = m - (potion_index + 1)
        return answer

    def distributeCookies(self, cookies: List[int], k: int) -> int:
        # 2305.Fair Distribution of Cookies
        cur = [0] * k
        n = len(cookies)

        def dfs(i, zero_count):
            # If there are not enough cookies remaining, return `float('inf')`
            # as it leads to an invalid distribution.
            if n - i < zero_count:
                return float("inf")
            # After distributing all cookies, return the unfairness of this
            # distribution.
            if i == n:
                return max(cur)
            # Try to distribute the i-th cookie to each child, and update answer
            # as the minimum unfairness in these distributions.
            answer = float("inf")
            for j in range(k):
                zero_count -= int(cur[j] == 0)
                cur[j] += cookies[i]
                # Recursively distribute the next cookie.
                answer = min(answer, dfs(i + 1, zero_count))
                cur[j] -= cookies[i]
                zero_count += int(cur[j] == 0)
            return answer

        return dfs(0, k)

    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        # 2316.Count Unreachable Pairs of Nodes in an Undirected Graph
        adj_list = defaultdict(set)
        for a, b in edges:
            adj_list[a].add(b)
            adj_list[b].add(a)
        # Find connected components using depth-first search
        checked = set()
        components = []
        for i in range(n):
            if i not in checked:
                component = set()
                stack = [i]
                while stack:
                    u = stack.pop()
                    checked.add(u)
                    component.add(u)
                    for v in adj_list[u]:
                        if v not in checked:
                            stack.append(v)
                components.append(component)
        # Count number of pairs in each connected component
        pairs = 0
        for component in components:
            nodes = len(component)
            pairs += (nodes * (nodes - 1)) // 2
        return (n * (n - 1)) // 2 - pairs
        # ===============================================
        # class UnionFind:
        #     def __init__(self, n):
        #         self.parent = list(range(n))
        #         self.size = [1] * n
        #
        #     def find(self, x):
        #         if self.parent[x] != x:
        #             self.parent[x] = self.find(self.parent[x])
        #         return self.parent[x]
        #
        #     def union(self, x, y):
        #         root_x, root_y = self.find(x), self.find(y)
        #         if root_x != root_y:
        #             if self.size[root_x] < self.size[root_y]:
        #                 root_x, root_y = root_y, root_x
        #             self.parent[root_y] = root_x
        #             self.size[root_x] += self.size[root_y]
        #
        # uf = UnionFind(n)
        # for a, b in edges:
        #     uf.union(a, b)
        # components_size = [uf.size[i] for i in range(n) if uf.find(i) == i]
        # total_unreachable = sum(x * (n - x) for x in components_size)
        # return total_unreachable // 2

    def countPaths(self, grid: List[List[int]]) -> int:
        # 2328.Number of Increasing Paths in a Grid
        m, n = len(grid), len(grid[0])
        mod = 10**9 + 7
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dp = [[0] * n for _ in range(m)]

        def dfs(x: int, y: int) -> int:
            if dp[x][y]:
                return dp[x][y]
            answer = 1
            for dx, dy in directions:
                prev_x, prev_y = x + dx, y + dy
                if 0 <= prev_x < m and 0 <= prev_y < n and grid[prev_x][prev_y] < grid[x][y]:
                    answer += dfs(prev_x, prev_y)
            dp[x][y] = answer
            return answer

        return sum(dfs(i, j) for i in range(m) for j in range(n)) % mod

    def zeroFilledSubarray(self, nums: List[int]) -> int:
        # 2348.Number of Zero-Filled Subarrays
        zero_subs = []
        lf = rf = -1
        for i, num in enumerate(nums):
            if num == 0 and lf == -1 and rf == -1:
                lf = rf = i
            elif num == 0:
                rf = i
            elif num != 0 and rf != -1:
                zero_subs.append(rf - lf + 1)
                lf = rf = -1
            if i == len(nums) - 1 and num == 0:
                zero_subs.append(rf - lf + 1)
        ans = 0
        for nzero in zero_subs:
            ans += ((1 + nzero) * nzero) // 2
        return ans

    def equalPairs(self, grid: List[List[int]]) -> int:
        # 2352.Equal Row and Column Pairs
        mdict: defaultdict = defaultdict(int)
        ans = 0
        for row in grid:
            mdict[tuple(row)] += 1
        for col in zip(*grid):
            ans += mdict[tuple(col)]
        return ans

    def longestCycle(self, edges: List[int]) -> int:
        # 2360.Longest Cycle in a Graph
        longest_cycle_len = -1
        time_step = 1
        node_visited_at_time = [0] * len(edges)
        for current_node in range(len(edges)):
            if node_visited_at_time[current_node] > 0:
                continue
            start_time = time_step
            u = current_node
            while u != -1 and node_visited_at_time[u] == 0:
                node_visited_at_time[u] = time_step
                time_step += 1
                u = edges[u]
            if u != -1 and node_visited_at_time[u] >= start_time:
                longest_cycle_len = max(longest_cycle_len, time_step - node_visited_at_time[u])
        return longest_cycle_len

    def validPartition(self, nums: List[int]) -> bool:
        # 2369.Check if There is a Valid Partition For The Array
        n = len(nums)
        # Only record the most recent 3 indices
        dp = [True] + [False] * 2
        # Determine if the prefix array nums[0 ~ i] has a valid partition
        for i in range(n):
            dp_index = i + 1
            ans = False
            if i > 0 and nums[i] == nums[i - 1]:
                ans |= dp[(dp_index - 2) % 3]
            if i > 1 and nums[i] == nums[i - 1] == nums[i - 2]:
                ans |= dp[(dp_index - 3) % 3]
            if i > 1 and nums[i] == nums[i - 1] + 1 == nums[i - 2] + 2:
                ans |= dp[(dp_index - 3) % 3]
            dp[dp_index % 3] = ans
        return dp[n % 3]

    def removeStars(self, s: str) -> str:
        # 2390.Removing Stars From a String
        stack: List[str] = []
        for c in s:
            if c == "*":
                stack.pop()
            else:
                stack.append(c)
        return "".join(stack)


class SmallestInfiniteSet:  # pragma: no cover
    # 2336.Smallest Number in Infinite Set
    def __init__(self):
        self.is_present = set()
        self.added_int = []
        self.curr_int = 1

    def popSmallest(self) -> int:
        if len(self.added_int):
            answer = heapq.heappop(self.added_int)
            self.is_present.remove(answer)
        else:
            answer = self.curr_int
            self.curr_int += 1
        return answer

    def addBack(self, num: int) -> None:
        if self.curr_int <= num or num in self.is_present:
            return
        heapq.heappush(self.added_int, num)
        self.is_present.add(num)
