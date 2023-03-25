from collections import defaultdict
from typing import List


class Pro2201To2400:
    def __init__(self) -> None:
        pass

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
