from collections import Counter, deque
from typing import List, Optional

from .utils import ListNode, TreeNode


class Pro0801To1000:
    def __init__(self):
        pass

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
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def sortArray(self, nums: List[int]) -> List[int]:
        # 912.Sort an Array
        nums_dict = Counter(nums)
        nums_sorted = []
        for num in range(min(nums), max(nums) + 1):
            nums_sorted.extend([num] * nums_dict[num])
        return nums_sorted

    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        # 958.Check Completeness of a Binary Tree
        queue = deque([root])
        while queue[0]:
            node = queue.popleft()
            queue.extend([node.left, node.right])
        while queue and not queue[0]:
            queue.popleft()
        if queue:
            return False
        return True

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

    def orangesRotting(self, grid: List[List[int]]) -> int:
        # 994.Rotting Oranges
        m, n = len(grid), len(grid[0])
        if all(grid[i][j] != 1 for i in range(m) for j in range(n)):
            return 0
        time = 0
        dx = [0, 1, 0, -1, 0]
        queue = deque([])
        queue_next = deque([])
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
