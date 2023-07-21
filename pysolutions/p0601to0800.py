import heapq
from collections import Counter
from typing import List, Optional

from .utils import ListNode, TreeNode


class Pro0601To0800:
    def __init__(self):
        pass

    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        # 605.Can Place Flowers
        def have_noadj(flowerbed, index):
            lf, rf = True, True
            if index > 0 and flowerbed[index - 1] == 1:
                lf = False
            if index < len(flowerbed) - 1 and flowerbed[index + 1] == 1:
                rf = False
            if lf and rf:
                return 1

        flowers = 0
        for i in range(len(flowerbed)):
            if flowerbed[i] == 0 and have_noadj(flowerbed, i):
                flowers += 1
                flowerbed[i] = 1
        return n <= flowers

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        # 617.Merge Two Binary Trees
        def merge_bin_tree(curr1: TreeNode, curr2: TreeNode):
            curr1.val += curr2.val
            if curr1.left is not None and curr2.left is not None:
                merge_bin_tree(curr1.left, curr2.left)
            elif curr1.left is None:
                curr1.left = curr2.left
            if curr1.right is not None and curr2.right is not None:
                merge_bin_tree(curr1.right, curr2.right)
            elif curr1.right is None:
                curr1.right = curr2.right

        if root1 is not None and root2 is not None:
            curr1: TreeNode = root1
            curr2: TreeNode = root2
            merge_bin_tree(curr1, curr2)
            return root1
        elif root1 is not None and root2 is None:
            return root1
        elif root1 is None and root2 is not None:
            return root2
        else:
            return None

    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 621.Task Scheduler
        ct_tasks = Counter(tasks)
        max_freq = max(ct_tasks.values())
        max_freq_tasks = [key for key, value in ct_tasks.items() if value == max_freq]
        return max(len(tasks), (max_freq - 1) * (n + 1) + len(max_freq_tasks))

    def judgeSquareSum(self, c: int) -> bool:
        # 633.Sum of Square Numbers
        a, b = 0, int(c**0.5)
        while a <= b:
            curr_sum = a**2 + b**2
            if curr_sum == c:
                return True
            elif curr_sum < c:
                a += 1
            else:
                b -= 1
        return False

    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 643.Maximum Average Subarray I
        max_sum = curr_sum = sum(nums[:k])
        for i in range(k, len(nums)):
            curr_sum += nums[i] - nums[i - k]
            max_sum = max(max_sum, curr_sum)
        return max_sum / k

    def predictPartyVictory(self, senate: str) -> str:
        # 649.Dota2 Senate
        radiant, dire = [], []
        for i, s in enumerate(senate):
            if s == "R":
                radiant.append(i)
            else:
                dire.append(i)
        while radiant and dire:
            if radiant[0] < dire[0]:
                radiant.append(radiant[0] + len(senate))
            else:
                dire.append(dire[0] + len(senate))
            radiant.pop(0)
            dire.pop(0)
        return "Radiant" if radiant else "Dire"

    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        # 653.Two Sum IV - Input is a BST
        if not root:
            return False
        tree_list = []
        queue: List[Optional[TreeNode]] = [root]
        while queue:
            node = queue.pop(0)
            if node:
                tree_list.append(node.val)
                queue.extend([node.left, node.right] if node else [])
            else:
                tree_list.append(None)
        while tree_list[-1] is None:
            tree_list.pop()
        checked = set()
        for num in tree_list:
            if num is None:
                continue
            elif k - num in checked:
                return True
            else:
                checked.add(num)
        return False

    def judgeCircle(self, moves: str) -> bool:
        # 657.Robot Return to Origin
        return moves.count("U") == moves.count("D") and moves.count("L") == moves.count("R")

    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # 662.Maximum Width of Binary Tree
        if not root:
            return 0
        queue = [(root, 0)]
        max_width = 0
        while queue:
            max_width = max(max_width, queue[-1][1] - queue[0][1] + 1)
            for _ in range(len(queue)):
                node, index = queue.pop(0)
                if node.left:
                    queue.append((node.left, index * 2))
                if node.right:
                    queue.append((node.right, index * 2 + 1))
        return max_width

    def findNumberOfLIS(self, nums: List[int]) -> int:
        # 673.Number of Longest Increasing Subsequence
        if not nums:
            return 0
        dp = [[1, 1] for _ in range(len(nums))]
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[i][0] < dp[j][0] + 1:
                        dp[i][0] = dp[j][0] + 1
                        dp[i][1] = dp[j][1]
                    elif dp[i][0] == dp[j][0] + 1:
                        dp[i][1] += dp[j][1]
        max_len = max(dp, key=lambda x: x[0])[0]
        return sum([x[1] for x in dp if x[0] == max_len])

    def findNumberOfLIS_v2(self, nums: List[int]) -> int:
        # 673.Number of Longest Increasing Subsequence
        n = len(nums)
        length = [0] * n
        count = [0] * n

        def calculate_dp(i):
            if length[i] != 0:
                return
            length[i] = 1
            count[i] = 1
            for j in range(i):
                if nums[j] < nums[i]:
                    calculate_dp(j)
                    if length[j] + 1 > length[i]:
                        length[i] = length[j] + 1
                        count[i] = 0
                    if length[j] + 1 == length[i]:
                        count[i] += count[j]

        max_length = 0
        result = 0
        for i in range(n):
            calculate_dp(i)
            max_length = max(max_length, length[i])
        for i in range(n):
            if length[i] == max_length:
                result += count[i]
        return result

    def calPoints(self, operations: List[str]) -> int:
        # 682.Baseball Game
        stack: List[int] = []
        for op in operations:
            if op == "+":
                stack.append(stack[-1] + stack[-2])
            elif op == "D":
                stack.append(stack[-1] * 2)
            elif op == "C":
                stack.pop()
            else:
                stack.append(int(op))
        return sum(stack)

    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # 692.Top K Frequent Words
        ct_words = Counter(sorted(words))
        freq_words = ct_words.most_common(k)
        return list(key for key, _ in freq_words)

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 695.Max Area of Island
        def dfs(grid, row, col):
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] == 0:
                return 0
            grid[row][col] = 0
            area = 1
            area += dfs(grid, row - 1, col)
            area += dfs(grid, row + 1, col)
            area += dfs(grid, row, col - 1)
            area += dfs(grid, row, col + 1)
            return area

        max_area = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == 1:
                    max_area = max(max_area, dfs(grid, row, col))
        return max_area

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # 700.Search in a Binary Search Tree
        if not root:
            return None
        elif root.val == val:
            return root
        elif root.val < val:
            return self.searchBST(root.right, val)
        else:
            return self.searchBST(root.left, val)

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # 701.Insert into a Binary Search Tree
        if not root:
            return TreeNode(val)
        elif root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root

    def search(self, nums: List[int], target: int) -> int:
        # 704.Binary Search
        def bin_search(nums: List[int], target: int, base: int):
            if len(nums) == 0:
                return -1
            indmid = len(nums) // 2
            if target == nums[indmid]:
                return base + indmid
            elif target > nums[indmid]:
                return bin_search(nums[indmid + 1 :], target, base + indmid + 1)
            else:
                return bin_search(nums[:indmid], target, base)

        return bin_search(nums, target, 0)

    def toLowerCase(self, s: str) -> str:
        # 709.To Lower Case
        ans = ""
        for alpha in s:
            ans += alpha.lower()
        return ans

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # 713.Subarray Product Less Than K
        if k <= 1:
            return 0
        prod = 1
        ans = left = 0
        for right, val in enumerate(nums):
            prod *= val
            while prod >= k:
                prod = int(prod / nums[left])
                left += 1
            ans += right - left + 1
        return ans

    def maxProfit(self, prices: List[int], fee: int) -> int:
        # 714.Best Time to Buy ans Sell Stock with Transaction Fee
        cash, hold = 0, -prices[0]
        for i in range(1, len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
        return cash

    def pivotIndex(self, nums: List[int]) -> int:
        # 724.Find Pivot Index
        total = sum(nums)
        lsum = 0
        for i in range(len(nums)):
            if total - nums[i] - lsum == lsum:
                return i
            lsum += nums[i]
        return -1

    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        # 733.Flood Fill
        def dfs(image, row, col, starting_color, color):
            if (
                row < 0
                or col < 0
                or row >= len(image)
                or col >= len(image[0])
                or image[row][col] != starting_color
                or image[row][col] == color
            ):
                return
            image[row][col] = color
            dfs(image, row - 1, col, starting_color, color)
            dfs(image, row + 1, col, starting_color, color)
            dfs(image, row, col - 1, starting_color, color)
            dfs(image, row, col + 1, starting_color, color)

        starting_color = image[sr][sc]
        if starting_color == color:
            return image
        dfs(image, sr, sc, starting_color, color)
        return image

    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # 735.Asteroid Collision
        stack: List[int] = []
        for asteroid in asteroids:
            while stack and asteroid < 0 < stack[-1]:
                if stack[-1] < -asteroid:
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    stack.pop()
                break
            else:
                stack.append(asteroid)
        return stack

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # 739.Daily Temperatures
        ans = [0] * len(temperatures)
        stack: List[int] = []
        for i, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temp:
                curr = stack.pop()
                ans[curr] = i - curr
            stack.append(i)
        return ans

    def deleteAndEarn(self, nums: List[int]) -> int:
        # 740.Delete and Earn
        max_val = max(nums)
        points = [0] * (max_val + 1)
        for num in nums:
            points[num] += num
        prev, curr = 0, 0
        for point in points:
            prev, curr = curr, max(prev + point, curr)
        return curr

    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        # 744.Find Smallest Letter Greater Than Target
        if letters[-1] <= target:
            return letters[0]
        left, right = 0, len(letters) - 1
        while left < right:
            mid = (left + right) // 2
            if letters[mid] > target:
                right = mid
            else:
                left = mid + 1
        return letters[left]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # 746.Min Cost Climbing Stairs
        if not cost or len(cost) == 1:
            return 0
        dp = [0] * len(cost)
        dp[0], dp[1] = cost[0], cost[1]
        for i in range(2, len(cost)):
            dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])
        return min(dp[-1], dp[-2])

    def partitionLabels(self, s: str) -> List[int]:
        # 763.Partition Labels
        last = {alpha: i for i, alpha in enumerate(s)}
        index = anchor = 0
        ans: List[int] = []
        for i, alpha in enumerate(s):
            index = max(index, last[alpha])
            if i == index:
                ans.append(i - anchor + 1)
                anchor = i + 1
        return ans

    def letterCasePermutation(self, s: str) -> List[str]:
        # 784.Letter Case Permutation
        result = [""]
        for char in s:
            n = len(result)
            if char.isalpha():
                for i in range(n):
                    result.append(result[i] + char.lower())
                    result[i] += char.upper()
            else:
                for i in range(n):
                    result[i] += char
        return result

    def letterCasePermutation_v2(self, s: str) -> List[str]:
        # 784.Letter Case Permutation
        count = Counter(s)
        nalpha = 0
        for key in count.keys():
            if key.isalpha():
                nalpha += count[key]

        def backtrack(path):
            if len(path) == nalpha:
                up_or_low.append(path[:])
                return
            for i in [0, 1]:
                path.append(i)
                backtrack(path)
                path.pop()

        up_or_low: List[List[str]] = []
        backtrack([])
        ans = []
        for up_low in up_or_low:
            snew, index = "", 0
            for alpha in s:
                if alpha.isalpha():
                    if up_low[index]:
                        snew += alpha.upper()
                    else:
                        snew += alpha.lower()
                    index += 1
                else:
                    snew += alpha
            ans.append(snew)
        return ans

    def isBipartite(self, graph: List[List[int]]) -> bool:
        # 785.Is Graph Bipartite?
        n = len(graph)
        color = [0] * n

        def dfs(node, c):
            if color[node]:
                return color[node] == c
            color[node] = c
            return all(dfs(nei, -c) for nei in graph[node])

        return all(dfs(node, 1) for node in range(n) if not color[node])

    def numTilings(self, n: int) -> int:
        # 790.Domino and Tromino Tiling
        if n < 3:
            return n
        dp = [0] * (n + 1)
        dp[0], dp[1], dp[2] = 1, 1, 2
        for i in range(3, n + 1):
            dp[i] = (2 * dp[i - 1] + dp[i - 3]) % (10**9 + 7)
        return dp[n]

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # 797. All Path From Source to Target
        target = len(graph) - 1
        paths, targets = [[0]], []
        while paths:
            path = paths.pop(0)
            edges = graph[path[-1]]
            for edge in edges:
                if edge == target:
                    targets.append(path + [edge])
                else:
                    paths = [path + [edge]] + paths
        return targets


class MyCircularQueue:  # pragma: no cover
    # 622.Design Circular Queue
    def __init__(self, k: int):
        self.queue = [0] * k
        self.head = self.tail = -1
        self.size = k

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        if self.isEmpty():
            self.head = 0
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = value
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        if self.head == self.tail:
            self.head = self.tail = -1
        else:
            self.head = (self.head + 1) % self.size
        return True

    def Front(self) -> int:
        return self.queue[self.head] if not self.isEmpty() else -1

    def Rear(self) -> int:
        return self.queue[self.tail] if not self.isEmpty() else -1

    def isEmpty(self) -> bool:
        return self.head == -1

    def isFull(self) -> bool:
        return (self.tail + 1) % self.size == self.head


class KthLargest:  # pragma: no cover
    # 703.Kth Largest Element in a Stream
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heappushpop(self.heap, val)
        return self.heap[0]


class MyHashSet:  # pragma: no cover
    # 705.Design HashSet
    def __init__(self):
        self.buckets = [[] for _ in range(1000)]

    def _hash(self, key: int) -> int:
        return key % len(self.buckets)

    def add(self, key: int) -> None:
        index = self._hash(key)
        for k in self.buckets[index]:
            if k == key:
                return
        self.buckets[index].append(key)

    def remove(self, key: int) -> None:
        index = self._hash(key)
        for i, k in enumerate(self.buckets[index]):
            if k == key:
                self.buckets[index].pop(i)
                return

    def contains(self, key: int) -> bool:
        index = self._hash(key)
        for k in self.buckets[index]:
            if k == key:
                return True
        return False


class MyHashMap:  # pragma: no cover
    # 706.Design HashMap
    def __init__(self):
        self.buckets = [[] for _ in range(1000)]

    def _hash(self, key: int) -> int:
        return key % len(self.buckets)

    def put(self, key: int, value: int) -> None:
        index = self._hash(key)
        for i, (k, _) in enumerate(self.buckets[index]):
            if k == key:
                self.buckets[index][i] = (key, value)
                return
        self.buckets[index].append((key, value))

    def get(self, key: int) -> int:
        index = self._hash(key)
        for k, v in self.buckets[index]:
            if k == key:
                return v
        return -1

    def remove(self, key: int) -> None:
        index = self._hash(key)
        for i, (k, _) in enumerate(self.buckets[index]):
            if k == key:
                self.buckets[index].pop(i)
                return


class MyLinkedList:  # pragma: no cover
    # 707.Design Linked List
    def __init__(self):
        self.head = None
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        current = self.head
        for _ in range(index):
            current = current.next
        return current.val

    def addAtHead(self, val: int) -> None:
        self.head = ListNode(val, self.head)
        self.size += 1

    def addAtTail(self, val: int) -> None:
        if not self.head:
            self.head = ListNode(val)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = ListNode(val)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        if index <= 0:
            self.addAtHead(val)
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            current.next = ListNode(val, current.next)
            self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        if index == 0:
            self.head = self.head.next
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            current.next = current.next.next
        self.size -= 1


class MyCalendar:  # pragma: no cover
    # 729.My Calendar I
    def __init__(self):
        self.calendar = []

    def book(self, start: int, end: int) -> bool:
        for s, e in self.calendar:
            if s < end and start < e:
                return False
        self.calendar.append((start, end))
        return True
