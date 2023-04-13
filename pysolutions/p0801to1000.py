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
