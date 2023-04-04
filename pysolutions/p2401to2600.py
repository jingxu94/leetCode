import math
from queue import Queue
from typing import List


class Pro2401To2600:
    def __init__(self) -> None:
        pass

    def partitionString(self, s: str) -> int:
        # 2405.Optimal Partition of String
        last_seen = [-1] * 26
        count = 1
        starting = 0
        for i in range(len(s)):
            if last_seen[ord(s[i]) - ord("a")] >= starting:
                count += 1
                starting = i
            last_seen[ord(s[i]) - ord("a")] = i
        return count

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        # 2492.Minimum Score of a Path Between Two Cities
        ans = math.inf
        gr: List[List[tuple]] = [[] for _ in range(n + 1)]
        for edge in roads:
            gr[edge[0]].append((edge[1], edge[2]))  # u-> {v, dis}
            gr[edge[1]].append((edge[0], edge[2]))  # v-> {u, dis}
        vis = [0] * (n + 1)
        q: Queue = Queue()
        q.put(1)
        vis[1] = 1
        while not q.empty():
            node = q.get()
            for v, dis in gr[node]:
                ans = min(ans, dis)
                if vis[v] == 0:
                    vis[v] = 1
                    q.put(v)
        return int(ans)

    def splitNum(self, num: int) -> int:
        # 2578.Split With Minimum Sum
        digits = list(int(n) for n in str(num))
        digits.sort()
        digits.reverse()
        power1, power2 = len(digits) // 2, len(digits) - len(digits) // 2
        ans = 0
        for _ in range(power1):
            ans += 10 ** (power2 - 1) * digits[-1]
            digits.pop()
            ans += 10 ** (power1 - 1) * digits[-1]
            digits.pop()
            power1, power2 = power1 - 1, power2 - 1
        if power2:
            ans += digits[0]
        return ans
