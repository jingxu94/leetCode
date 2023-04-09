from collections import Counter, defaultdict
from typing import List


class Pro1801To2000:
    def __init__(self) -> None:
        pass

    def arraySign(self, nums: List[int]) -> int:
        # 1822.Sign of the Product of an Array
        if 0 in nums:
            return 0
        ct_num = Counter(nums)
        under_zero = 0
        for key in ct_num.keys():
            if key < 0:
                under_zero += ct_num[key]
        if under_zero % 2 == 1:
            return -1
        return 1

    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        # 1855.Maximum Distance Between a Pair of Values
        i, j = 0, 0
        max_dist = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                max_dist = max(max_dist, j - i)
                j += 1
            else:
                i += 1
                j = max(j, i)
        return max_dist

    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        # 1857.Largest Color Value in a Directed Graph
        n = len(colors)
        adj_list = defaultdict(list)
        in_degrees = [0] * n
        # Create the adjacency list and calculate in-degrees
        for u, v in edges:
            adj_list[u].append(v)
            in_degrees[v] += 1
        # Topological sorting using BFS
        queue = [i for i in range(n) if in_degrees[i] == 0]
        topo_order = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adj_list[node]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)
        # If there's a cycle in the graph, return -1
        if len(topo_order) != n:
            return -1
        # Calculate the largest color value along the paths
        color_count = [[0] * 26 for _ in range(n)]
        for node in topo_order:
            color = colors[node]
            color_count[node][ord(color) - ord("a")] += 1
            for neighbor in adj_list[node]:
                for i in range(26):
                    color_count[neighbor][i] = max(color_count[neighbor][i], color_count[node][i])
        return max(max(row) for row in color_count)

    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        # 1886.Determine Whether Matrix Can Be Obtained By Rotation
        if mat == target:
            return True

        def rotate(mat: List[List[int]]):
            return [list(x) for x in zip(*mat)][::-1]

        for _ in range(3):
            mat = rotate(mat)
            if mat == target:
                return True
        return False
