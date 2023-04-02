from collections import defaultdict
from typing import List


class Pro2201To2400:
    def __init__(self) -> None:
        pass

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
