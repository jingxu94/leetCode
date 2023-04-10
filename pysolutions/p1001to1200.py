from collections import Counter, deque
from typing import List


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

    def lastStoneWeight(self, stones: List[int]) -> int:
        # 1046.Last Stone Weight
        while len(stones) > 1:
            stones.sort()
            y = stones.pop()
            x = stones.pop()
            if y != x:
                stones.append(y - x)
        return stones[0] if stones else 0

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
