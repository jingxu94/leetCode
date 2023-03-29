from collections import Counter, deque
from typing import List, Optional

from .utils import ListNode, TreeNode


class Pro0801To1000:
    def __init__(self):
        pass

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

    def isMonotonic(self, nums: List[int]) -> bool:
        # 896.Monotonic Array
        if nums == sorted(nums):
            return True
        else:
            nums.reverse()
            return True if nums == sorted(nums) else False

    def sortArray(self, nums: List[int]) -> List[int]:
        # 912.Sort an Array
        nums_dict = Counter(nums)
        nums_sorted = []
        for num in range(min(nums), max(nums) + 1):
            nums_sorted.extend([num] * nums_dict[num])
        return nums_sorted

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
