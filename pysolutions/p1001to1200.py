import bisect
import math

from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional
from .utils import TreeNode


class Pro1001To1200:
    def __init__(self):
        pass

    def commonChars(self, words: List[str]) -> List[str]:
        # 1002.Find Common Characters
        flag = Counter(words[0])
        if len(words) == 1:
            return [i for i in flag.elements()]
        for word in words[1:]:
            check = Counter(word)
            for key in flag.keys():
                if check.get(key, 0) < flag[key]:
                    flag[key] = check[key]
        return [i for i in flag.elements()]

    def longestOnes(self, nums: List[int], k: int) -> int:
        # 1004.Max Consecutive Ones III
        left = right = 0
        ans = 0
        while right < len(nums):
            if nums[right] == 0:
                k -= 1
            while k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1
            ans = max(ans, right - left + 1)
            right += 1
        return ans

    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        # 1014.Best Sightseeing Pair
        ans = imax = 0
        for i, val in enumerate(values):
            ans = max(ans, imax + val - i)
            imax = max(imax, val + i)
        return ans

    def numEnclaves(self, grid: List[List[int]]) -> int:
        # 1020.Number of Enclaves
        m, n = len(grid), len(grid[0])

        def dfs(grid, row, col):
            if row < 0 or row >= m or col < 0 or col >= n or grid[row][col] == 0:
                return
            grid[row][col] = 0
            dfs(grid, row - 1, col)
            dfs(grid, row + 1, col)
            dfs(grid, row, col - 1)
            dfs(grid, row, col + 1)

        for i in range(m):
            dfs(grid, i, 0)
            dfs(grid, i, n - 1)
        for j in range(n):
            dfs(grid, 0, j)
            dfs(grid, m - 1, j)
        return len(list(1 for i in range(m) for j in range(n) if grid[i][j] == 1))

    def longestArithSeqLength(self, nums: List[int]) -> int:
        # 1027.Longest Arithmetic Subsequence
        dp: Dict[int, Dict[int, int]] = defaultdict(dict)
        ans = 0
        for i in range(len(nums)):
            for j in range(i):
                diff = nums[i] - nums[j]
                dp[i][diff] = dp[j].get(diff, 1) + 1
                ans = max(ans, dp[i][diff])
        return ans

    def isRobotBounded(self, instructions: str) -> bool:
        # 1041.Robot Bounded In Circle
        x, y, dx, dy = 0, 0, 0, 1
        for i in instructions:
            if i == "G":
                x, y = x + dx, y + dy
            elif i == "L":
                dx, dy = -dy, dx
            else:
                dx, dy = dy, -dx
        return (x, y) == (0, 0) or (dx, dy) != (0, 1)

    def lastStoneWeight(self, stones: List[int]) -> int:
        # 1046.Last Stone Weight
        while len(stones) > 1:
            stones.sort()
            y = stones.pop()
            x = stones.pop()
            if y != x:
                stones.append(y - x)
        return stones[0] if stones else 0

    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        # 1035.Uncrossed Lines
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for row in range(1, m + 1):
            for col in range(1, n + 1):
                if nums1[row - 1] == nums2[col - 1]:
                    dp[row][col] = dp[row - 1][col - 1] + 1
                else:
                    dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
        return dp[m][n]

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        # 1071.Greatest Common Divisor of Strings
        if str1 + str2 != str2 + str1:
            return ""
        else:
            return str1[: math.gcd(len(str1), len(str2))]

    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        # 1091.Shortest Path in Binary Matrix
        row, col = len(grid), len(grid[0])
        queue: deque = deque([(0, 0, 1)])
        if grid[0][0] == 1:
            return -1
        while queue:
            x, y, steps = queue.popleft()
            if x == row - 1 and y == col - 1:
                return steps
            for nx, ny in [
                [x + 1, y + 1],
                [x - 1, y - 1],
                [x + 1, y - 1],
                [x - 1, y + 1],
                [x + 1, y],
                [x - 1, y],
                [x, y + 1],
                [x, y - 1],
            ]:
                if 0 <= nx < row and 0 <= ny < col and grid[nx][ny] == 0:
                    grid[nx][ny] = 2
                    queue.append((nx, ny, steps + 1))
        return -1

    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        # 1125.Smallest Sufficient Team
        n = len(req_skills)
        key = {v: i for i, v in enumerate(req_skills)}
        dp: Dict[int, List[int]] = {0: []}
        for i, p in enumerate(people):
            his_skill = 0
            for skill in p:
                his_skill |= 1 << key[skill]
            for skill_set, need in list(dp.items()):
                with_him = skill_set | his_skill
                if with_him == skill_set:
                    continue
                if with_him not in dp or len(dp[with_him]) > len(need) + 1:
                    dp[with_him] = need + [i]
        return dp[(1 << n) - 1]

    def tribonacci(self, n: int) -> int:
        # 1137.N-th Tribonacci Number
        if n == 0 or n == 1:
            return n
        elif n == 2:
            return 1
        else:
            fst, sec, trd = 0, 1, 1
            for _ in range(n - 2):
                fst, sec, trd = sec, trd, fst + sec + trd
            return trd

    def stoneGameII(self, piles: List[int]) -> int:
        # 1140.Stone Game II
        n = len(piles)
        for i in range(n - 2, -1, -1):
            piles[i] += piles[i + 1]
        memo: Dict[set, int] = {}

        def helper(i, m):
            if (i, m) in memo:
                return memo[(i, m)]
            if i + 2 * m >= n:
                return piles[i]
            best = 0
            for x in range(1, 2 * m + 1):
                best = max(best, piles[i] - helper(i + x, max(m, x)))
            memo[(i, m)] = best
            return best

        return helper(0, 1)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 1143.Longest Common Subsequence
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        # 1161.Maximum Level Sum of a Binary Tree
        queue: deque = deque([(root, 1)])
        level = 1
        ans = 0
        max_sum = int(-1e5)
        while queue:
            level_sum = 0
            for _ in range(len(queue)):
                node, _ = queue.popleft()
                level_sum += node.val
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))
            if level_sum > max_sum:
                max_sum = level_sum
                ans = level
            level += 1
        return ans

    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        # 1187.Make Array Strictly Increasing
        arr2.sort()
        dp = {-1: 0}
        for num in arr1:
            tmp: defaultdict = defaultdict(lambda: float("inf"))
            for key in dp:
                if num > key:
                    tmp[num] = min(tmp[num], dp[key])
                loc = bisect.bisect_right(arr2, key)
                if loc < len(arr2):
                    tmp[arr2[loc]] = min(tmp[arr2[loc]], dp[key] + 1)
            dp = tmp
        if dp:
            return min(dp.values())
        else:
            return -1


class SnapshotArray:  # pragma: no cover
    # 1146.Snapshot Array
    def __init__(self, length: int):
        self.id = 0
        self.history_records = [[[0, 0]] for _ in range(length)]

    def set(self, index: int, val: int) -> None:
        self.history_records[index].append([self.id, val])

    def snap(self) -> int:
        self.id += 1
        return self.id - 1

    def get(self, index: int, snap_id: int) -> int:
        history_record = self.history_records[index]
        left, right = 0, len(history_record) - 1
        while left <= right:
            mid = (left + right) // 2
            if history_record[mid][0] <= snap_id:
                left = mid + 1
            else:
                right = mid - 1
        return history_record[right][1]
