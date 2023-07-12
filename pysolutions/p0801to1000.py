from collections import Counter, defaultdict, deque
from typing import List, Optional

from .utils import ListNode, TreeNode


class Pro0801To1000:
    def __init__(self):
        pass

    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        # 802.Find Eventual Safe States
        n = len(graph)
        indegree = [0] * n
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for node in graph[i]:
                adj[node].append(i)
                indegree[i] += 1
        q: deque[int] = deque()
        # Push all the nodes with indegree zero in the queue.
        for i in range(n):
            if indegree[i] == 0:
                q.append(i)
        safe = [False] * n
        while q:
            node = q.popleft()
            safe[node] = True
            for neighbor in adj[node]:
                # Delete the edge "node -> neighbor".
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    q.append(neighbor)
        safeNodes = []
        for i in range(n):
            if safe[i]:
                safeNodes.append(i)
        return safeNodes

    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        # 815.Bus Routes
        if source == target:
            return 0
        # Create a dictionary to store bus stops and the buses that stop there
        stop_to_buses = defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                stop_to_buses[stop].add(i)
        # Initialize the BFS queue and visited sets
        queue = deque([(source, 0)])  # (stop, number_of_buses_taken)
        visited_stops = {source}
        visited_buses = set()
        while queue:
            current_stop, buses_taken = queue.popleft()
            # Check all buses that stop at the current stop
            for bus in stop_to_buses[current_stop]:
                # If the bus has not been visited before
                if bus not in visited_buses:
                    visited_buses.add(bus)
                    # Check all stops on the bus route
                    for next_stop in routes[bus]:
                        # If the target stop is found, return the number of buses taken
                        if next_stop == target:
                            return buses_taken + 1
                        # If the next stop has not been visited, add it to the queue
                        if next_stop not in visited_stops:
                            visited_stops.add(next_stop)
                            queue.append((next_stop, buses_taken + 1))
        # If the target is not reachable, return -1
        return -1

    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        # 837.New 21 Game
        dp = [0.0] * (n + 1)
        dp[0] = 1.0
        s = 1.0 if k > 0 else 0
        for i in range(1, n + 1):
            dp[i] = s / maxPts
            if i < k:
                s += dp[i]
            if i - maxPts >= 0 and i - maxPts < k:
                s -= dp[i - maxPts]
        return sum(dp[k:])

    def numSimilarGroups(self, strs: List[str]) -> int:
        # 839.Similar String Groups
        def is_similar(str1: str, str2: str) -> bool:
            count = 0
            for i in range(len(str1)):
                if str1[i] != str2[i]:
                    count += 1
            return count == 2 or count == 0

        def dfs(strs: List[str], visited: List[bool], index: int):
            visited[index] = True
            for i in range(len(strs)):
                if not visited[i] and is_similar(strs[index], strs[i]):
                    dfs(strs, visited, i)

        visited = [False for _ in range(len(strs))]
        count = 0
        for i in range(len(strs)):
            if not visited[i]:
                dfs(strs, visited, i)
                count += 1
        return count

    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        # 841.Keys and Rooms
        visited = set()
        queue = deque([0])
        while queue:
            room = queue.popleft()
            visited.add(room)
            for key in rooms[room]:
                if key not in visited:
                    queue.append(key)
        return len(visited) == len(rooms)

    def backspaceCompare(self, s: str, t: str) -> bool:
        # 844.Backspace String Compare
        stack_s, stack_t = "", ""
        for i in range(len(s)):
            if s[i] == "#":
                stack_s = stack_s[:-1]
            else:
                stack_s += s[i]
        for i in range(len(t)):
            if t[i] == "#":
                stack_t = stack_t[:-1]
            else:
                stack_t += t[i]
        return stack_s == stack_t

    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 852.Peak Index in a Mountain Array
        left, right = 0, len(arr) - 1
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < arr[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    def buddyStrings(self, s: str, goal: str) -> bool:
        # 859.Buddy Strings
        if len(s) != len(goal):
            return False
        if s == goal:
            return len(set(s)) < len(s)
        diff = []
        for i in range(len(s)):
            if s[i] != goal[i]:
                diff.append(i)
        return len(diff) == 2 and s[diff[0]] == goal[diff[1]] and s[diff[1]] == goal[diff[0]]

    def lemonadeChange(self, bills: List[int]) -> bool:
        # 860.Lemonade Change
        five, ten = 0, 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                five -= 1
                ten += 1
            elif ten > 0:
                ten -= 1
                five -= 1
            else:
                five -= 3
            if five < 0:
                return False
        return True

    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # 863.All Nodes Distance K in Binary Tree
        graph = defaultdict(list)

        # Recursively build the undirected graph from the given binary tree.
        def build_graph(cur, parent):
            if cur and parent:
                graph[cur.val].append(parent.val)
                graph[parent.val].append(cur.val)
            if cur.left:
                build_graph(cur.left, cur)
            if cur.right:
                build_graph(cur.right, cur)

        build_graph(root, None)
        answer = []
        visited = set([target.val])
        # Add the target node to the queue with a distance of 0
        queue = deque([(target.val, 0)])
        while queue:
            cur, distance = queue.popleft()
            # If the current node is at distance k from target,
            # add it to the answer list and continue to the next node.
            if distance == k:
                answer.append(cur)
                continue
            # Add all unvisited neighbors of the current node to the queue.
            for neighbor in graph[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return answer

    def shortestPathAllKeys(self, grid: List[str]) -> int:
        # 864.Shortest Path to Get All Keys
        m = len(grid)
        n = len(grid[0])
        arr: deque = deque([])
        numOfKeys = 0
        keys = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
        locks = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "@":
                    arr.append((i, j, 0, 0))
                elif grid[i][j] in keys:
                    numOfKeys += 1
        visited = set()
        while arr:
            size = len(arr)
            for _ in range(size):
                i, j, found, steps = arr.popleft()
                curr = grid[i][j]
                if curr in locks and not (found >> locks[curr]) & 1 or curr == "#":
                    continue
                if curr in keys:
                    found |= 1 << keys[curr]
                    if found == (1 << numOfKeys) - 1:
                        return steps
                for x, y in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
                    if 0 <= x < m and 0 <= y < n and (x, y, found) not in visited:
                        visited.add((x, y, found))
                        arr.append((x, y, found, steps + 1))
        return -1

    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        # 872.Leaf-Similar Trees
        def get_leaves(root: Optional[TreeNode]) -> List[int]:
            if not root:
                return []
            if not root.left and not root.right:
                return [root.val]
            return get_leaves(root.left) + get_leaves(root.right)

        return get_leaves(root1) == get_leaves(root2)

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 875.Koko Eating Bananas
        def eat(piles: List[int], h: int, speed: int):
            time_list = []
            for pile in piles:
                time = pile // speed
                if pile % speed:
                    time += 1
                time_list.append(time)
            return sum(time_list) <= h

        left, right = 1, max(piles)
        result = -1
        while left <= right:
            speed = int((left + right) / 2)
            if eat(piles, h, speed):
                right = speed - 1
                result = speed
            else:
                left = speed + 1
        return result

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 876.Middle of the Linked List
        if not head:
            return None
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        # 879.Profitable Schemes
        mod = 10**9 + 7
        dp = [[[0 for _ in range(minProfit + 1)] for _ in range(n + 1)] for _ in range(len(group) + 1)]
        for count in range(n + 1):
            dp[len(group)][count][minProfit] = 1
        for index in range(len(group) - 1, -1, -1):
            for count in range(n + 1):
                for prof in range(minProfit + 1):
                    dp[index][count][prof] = dp[index + 1][count][prof]
                    if count + group[index] <= n:
                        dp[index][count][prof] = (
                            dp[index][count][prof]
                            + dp[index + 1][count + group[index]][min(minProfit, prof + profit[index])]
                        ) % mod
        return dp[0][0][0]

    def numRescueBoats(self, people: List[int], limit: int) -> int:
        # 881.Boats to Save People
        people.sort()
        queue = deque(people)
        count: int = 0
        while queue:
            max_people = queue.pop()
            count += 1
            if queue:
                if queue[0] + max_people <= limit:
                    queue.popleft()
        return count

    def isMonotonic(self, nums: List[int]) -> bool:
        # 896.Monotonic Array
        if nums == sorted(nums):
            return True
        else:
            nums.reverse()
            return True if nums == sorted(nums) else False

    def smallestRangeII(self, nums: List[int], k: int) -> int:
        # 910.Smallest Range II
        nums.sort()
        result = nums[-1] - nums[0]
        for i in range(len(nums) - 1):
            max_num = max(nums[-1], nums[i] + 2 * k)
            min_num = min(nums[i + 1], nums[0] + 2 * k)
            result = min(result, max_num - min_num)
        return result

    def sortArray(self, nums: List[int]) -> List[int]:
        # 912.Sort an Array
        nums_dict = Counter(nums)
        nums_sorted = []
        for num in range(min(nums), max(nums) + 1):
            nums_sorted.extend([num] * nums_dict[num])
        return nums_sorted

    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        # 918.Maximum Sum Circular Subarray
        n = len(nums)
        # Find the max subarray sum without circular property
        max_sum = float("-inf")
        cur_sum = 0
        for i in range(n):
            cur_sum = max(nums[i], cur_sum + nums[i])
            max_sum = max(max_sum, cur_sum)
        # If all elements are negative, return the highest negative number
        if max_sum < 0:
            return int(max_sum)
        # Find the max subarray sum with circular property
        total_sum = sum(nums)
        cur_sum = 0
        # Invert the signs of the array elements
        for i in range(n):
            nums[i] = -nums[i]
        for i in range(n):
            cur_sum = max(nums[i], cur_sum + nums[i])
            max_sum = max(max_sum, total_sum + cur_sum)
        return int(max_sum)

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        # 931.Minimum Falling Path Sum
        n = len(matrix)
        for i in range(1, n):
            for j in range(n):
                if j == 0:
                    matrix[i][j] += min(matrix[i - 1][j], matrix[i - 1][j + 1])
                elif j == n - 1:
                    matrix[i][j] += min(matrix[i - 1][j], matrix[i - 1][j - 1])
                else:
                    matrix[i][j] += min(matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i - 1][j + 1])
        return min(matrix[-1])

    def shortestBridge(self, grid: List[List[int]]) -> int:
        # 934.Shortest Bridge
        n = len(grid)
        first_x, first_y = -1, -1
        # Find any land cell, and we treat it as a cell of island A.
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    first_x, first_y = i, j
                    break
        # bfsQueue for BFS on land cells of island A; secondBfsQueue for BFS on water cells.
        bfs_queue = [(first_x, first_y)]
        second_bfs_queue = [(first_x, first_y)]
        grid[first_x][first_y] = 2
        # BFS for all land cells of island A and add them to second_bfs_queue.
        while bfs_queue:
            new_bfs = []
            for x, y in bfs_queue:
                for cur_x, cur_y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                    if 0 <= cur_x < n and 0 <= cur_y < n and grid[cur_x][cur_y] == 1:
                        new_bfs.append((cur_x, cur_y))
                        second_bfs_queue.append((cur_x, cur_y))
                        grid[cur_x][cur_y] = 2
            bfs_queue = new_bfs
        distance = 0
        while second_bfs_queue:
            new_bfs = []
            for x, y in second_bfs_queue:
                for cur_x, cur_y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                    if 0 <= cur_x < n and 0 <= cur_y < n:
                        if grid[cur_x][cur_y] == 1:
                            return distance
                        elif grid[cur_x][cur_y] == 0:
                            new_bfs.append((cur_x, cur_y))
                            grid[cur_x][cur_y] = -1
            # Once we finish one round without finding land cells of island B, we will
            # start the next round on all water cells that are 1 cell further away from
            # island A and increment the distance by 1.
            second_bfs_queue = new_bfs
            distance += 1
        return distance  # pragma: no cover

    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        # 946.Validate Stack Sequences
        stack: List[int] = []
        i = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1
        return not stack

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        # 953.Verifying an Alien Dictionary
        if len(words) == 1:
            return True
        aldict = dict()
        for i, alpha in enumerate(order):
            aldict[alpha] = i

        def is_words_sorted(word1: str, word2: str) -> bool:
            for i in range(len(word1)):
                if i == len(word2):
                    return False
                elif aldict[word1[i]] == aldict[word2[i]]:
                    continue
                elif aldict[word1[i]] < aldict[word2[i]]:
                    return True
                else:
                    return False
            return True

        for i in range(len(words) - 1):
            if not is_words_sorted(words[i], words[i + 1]):
                return False
        return True

    def tallestBillboard(self, rods: List[int]) -> int:
        # 956.Tallest Billboard
        dp = {0: 0}
        for r in rods:
            # dp.copy() means we don't add r to these stands.
            new_dp = dp.copy()
            for diff, taller in dp.items():
                shorter = taller - diff
                # Add r to the taller stand, thus the height difference is increased to diff + r.
                new_dp[diff + r] = max(new_dp.get(diff + r, 0), taller + r)
                # Add r to the shorter stand, the height difference is expressed as abs(shorter + r - taller).
                new_diff = abs(shorter + r - taller)
                new_taller = max(shorter + r, taller)
                new_dp[new_diff] = max(new_dp.get(new_diff, 0), new_taller)
            dp = new_dp
        # Return the maximum height with 0 difference.
        return dp.get(0, 0)

    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        # 958.Check Completeness of a Binary Tree
        if not root:
            return True
        queue = deque([root])
        while queue[0]:
            node = queue.popleft()
            queue.extend([node.left, node.right])
        while queue and not queue[0]:
            queue.popleft()
        if queue:
            return False
        return True

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # 973.K Closest Points to Origin
        distances = list((point[0] ** 2 + point[1] ** 2, i) for i, point in enumerate(points))
        distances.sort()
        ans: List[List[int]] = []
        for i in range(k):
            ans.append(points[distances[i][1]])
        return ans

    def largestPerimeter(self, nums: List[int]) -> int:
        # 976.Largest Perimeter Triangle
        nums.sort()
        for i in range(len(nums) - 3, -1, -1):
            if nums[i] + nums[i + 1] > nums[i + 2]:
                return nums[i] + nums[i + 1] + nums[i + 2]
        return 0

    def sortedSquares(self, nums: List[int]) -> List[int]:
        # 977.Squares of a Sorted Array
        ans = []
        for num in nums:
            ans.append(num**2)
        return sorted(ans)

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        # 983.Minimum Cost For Tickets
        dp = [0] * 366
        day_set = set(days)
        for day in range(1, 366):
            if day not in day_set:
                dp[day] = dp[day - 1]
                continue
            dp[day] = min(
                dp[max(day - 1, 0)] + costs[0], dp[max(day - 7, 0)] + costs[1], dp[max(day - 30, 0)] + costs[2]
            )
        return dp[-1]

    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        # 986.Interval List Intersections
        i, j = 0, 0
        result = []
        while i < len(firstList) and j < len(secondList):
            start1, end1 = firstList[i]
            start2, end2 = secondList[j]
            # Calculate the intersection
            start = max(start1, start2)
            end = min(end1, end2)
            # Check if there is an intersection and add it to the result list
            if start <= end:
                result.append([start, end])
            # Move the pointer of the list with the smaller endpoint
            if end1 < end2:
                i += 1
            else:
                j += 1
        return result

    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        # 989.Add to Array-Form of Integer
        ans: List[int] = []
        addone = 0
        while k or num or addone:
            number = num.pop() if num else 0
            knum, k = k % 10, k // 10
            numans = number + knum + addone
            ans.append(numans % 10)
            addone = numans // 10
        ans.reverse()
        return ans

    def orangesRotting(self, grid: List[List[int]]) -> int:
        # 994.Rotting Oranges
        m, n = len(grid), len(grid[0])
        if all(grid[i][j] != 1 for i in range(m) for j in range(n)):
            return 0
        time = 0
        dx = [0, 1, 0, -1, 0]
        queue: deque = deque([])
        queue_next: deque = deque([])
        for row in range(m):
            for col in range(n):
                if grid[row][col] == 2:
                    queue.append((row, col))
        while queue:
            row, col = queue.popleft()
            for i in range(4):
                nr, nc = row + dx[i], col + dx[i + 1]
                if nr < 0 or nr == m or nc < 0 or nc == n or grid[nr][nc] != 1:
                    continue
                grid[nr][nc] = 2
                queue_next.append((nr, nc))
            if not queue:
                time += 1
                if queue_next:
                    queue, queue_next = queue_next, deque([])
                else:
                    time -= 1
        if any(grid[i][j] == 1 for i in range(m) for j in range(n)):
            return -1
        else:
            return time

    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        # 997.Find the Town Judge
        if n == 1:
            return 1
        trust_dict = dict()
        for i in range(1, n + 1):
            trust_dict[i] = 0
        for i in range(len(trust)):
            trust_dict[trust[i][0]] -= 1
            trust_dict[trust[i][1]] += 1
        for i in range(1, n + 1):
            if trust_dict[i] == n - 1:
                return i
        return -1


class StockSpanner:  # pragma: no cover
    # 901.Online Stock Span
    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        ans = 1
        while self.stack and self.stack[-1][0] <= price:
            ans += self.stack.pop()[1]
        self.stack.append((price, ans))
        return ans


class RecentCounter:  # pragma: no cover
    # 933.Number of Recent Calls
    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        self.queue.append(t)
        while self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)
