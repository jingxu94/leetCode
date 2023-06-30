import heapq

from collections import Counter, defaultdict
from typing import List


class Pro1801To2000:
    def __init__(self) -> None:
        pass

    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        # 1802.Maximum Value at a Given Index in a Bounded Array
        def getSum(index: int, value: int, n: int) -> int:
            count = 0
            if value > index:
                count += (value + value - index) * (index + 1) // 2
            else:
                count += (value + 1) * value // 2 + index - value + 1
            if value >= n - index:
                count += (value + value - n + 1 + index) * (n - index) // 2
            else:
                count += (value + 1) * value // 2 + n - index - value
            return count - value

        left, right = 1, maxSum
        while left < right:
            mid = (left + right + 1) // 2
            if getSum(index, mid, n) <= maxSum:
                left = mid
            else:
                right = mid - 1
        return left

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

    def findTheWinner(self, n: int, k: int) -> int:
        # 1823.Find the Winner of the Circular Game
        def findWinner(n: int, k: int) -> int:
            if n == 1:
                return 1
            return (findWinner(n - 1, k) + k - 1) % n + 1

        return findWinner(n, k)

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

    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        # 1926.Nearest Exit from Entrance in Maze
        m, n = len(maze), len(maze[0])
        queue = [(entrance[0], entrance[1], 0)]
        visited = set()
        while queue:
            x, y, dist = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if (x, y) != (entrance[0], entrance[1]) and (x == 0 or x == m - 1 or y == 0 or y == n - 1):
                return dist
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and maze[nx][ny] == ".":
                    queue.append((nx, ny, dist + 1))
        return -1

    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        # 1964.Find the Longest Valid Obstacle Course at Each Position
        stack: List[int] = []
        res: List[int] = []
        for obstacle in obstacles:
            if not stack or stack[-1] <= obstacle:
                stack.append(obstacle)
                res.append(len(stack))
            else:
                left, right = 0, len(stack) - 1
                while left < right:
                    mid = (left + right) // 2
                    if stack[mid] <= obstacle:
                        left = mid + 1
                    else:
                        right = mid
                stack[left] = obstacle
                res.append(left + 1)
        return res

    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        # 1970.Last Day Where You Can Still Cross
        dsu = DSU(row * col + 2)
        grid = [[0] * col for _ in range(row)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for i in range(row * col):
            r, c = cells[i][0] - 1, cells[i][1] - 1
            grid[r][c] = 1
            index_1 = r * col + c + 1
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                index_2 = new_r * col + new_c + 1
                if 0 <= new_r < row and 0 <= new_c < col and grid[new_r][new_c] == 1:
                    dsu.union(index_1, index_2)
            if c == 0:
                dsu.union(0, index_1)
            if c == col - 1:
                dsu.union(row * col + 1, index_1)
            if dsu.find(0) == dsu.find(row * col + 1):
                return i
        return -1


class DSU:  # pragma: no cover
    def __init__(self, n):
        self.root = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.root[x] != x:
            self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        if self.size[root_x] > self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] = self.size[root_x]


class SeatManager:  # pragma: no cover
    # 1845.Seat Reservation Manager
    def __init__(self, n: int):
        self.heap = list(range(1, n + 1))

    def reserve(self) -> int:
        return heapq.heappop(self.heap)

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.heap, seatNumber)
