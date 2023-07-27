from typing import List, Optional
from .utils import ListNode
from collections import defaultdict, deque


class Pro2001To2200:
    def __init__(self):
        pass

    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        # 2024.Maximize the Confusion of an Exam
        n = len(answerKey)
        left = 0
        right = 0
        max_len = 0
        count: defaultdict = defaultdict(int)
        while right < n:
            count[answerKey[right]] += 1
            while count["T"] > k and count["F"] > k:
                count[answerKey[left]] -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
            right += 1
        return max_len

    def getAverages(self, nums: List[int], k: int) -> List[int]:
        # 2090.K Radius Subarray Averages
        averages = [-1] * len(nums)
        # When a single element is considered then its average will be the number itself only.
        if k == 0:
            return nums
        n = len(nums)
        # Any index will not have 'k' elements in it's left and right.
        if 2 * k + 1 > n:
            return averages
        # First get the sum of first window of the 'nums' arrray.
        window_sum = sum(nums[: 2 * k + 1])
        averages[k] = window_sum // (2 * k + 1)
        # Iterate on rest indices which have at least 'k' elements
        # on its left and right sides.
        for i in range(2 * k + 1, n):
            # We remove the discarded element and add the new element to get current window sum.
            # 'i' is the index of new inserted element, and
            # 'i - (window size)' is the index of the last removed element.
            window_sum = window_sum - nums[i - (2 * k + 1)] + nums[i]
            averages[i - k] = window_sum // (2 * k + 1)
        return averages

    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 2095.Delete the Middle Node of a Linked List
        if not head or not head.next:
            return None
        slow = head
        fast = head.next.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        slow.next = slow.next.next
        return head

    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        # 2101.Detonate the Maximum Bombs
        graph = defaultdict(list)
        n = len(bombs)
        # Build the graph
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                xi, yi, ri = bombs[i]
                xj, yj, _ = bombs[j]
                # Create a path from node i to node j, if bomb i detonates bomb j.
                if ri**2 >= (xi - xj) ** 2 + (yi - yj) ** 2:
                    graph[i].append(j)

        def bfs(i):
            queue = deque([i])
            visited = set([i])
            while queue:
                cur = queue.popleft()
                for neib in graph[cur]:
                    if neib not in visited:
                        visited.add(neib)
                        queue.append(neib)
            return len(visited)

        answer = 0
        for i in range(n):
            answer = max(answer, bfs(i))
        return answer

    def pairSum(self, head: Optional[ListNode]) -> int:
        # 2130.Maximum Twin Sum of a Linked List
        if not head or not head.next:
            return 0
        slow, fast = head, head
        answer = 0
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        curr: Optional[ListNode] = slow
        prev: Optional[ListNode] = None
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        start = head
        while prev:
            answer = max(answer, start.val + prev.val)
            start = start.next
            prev = prev.next
        return answer

    def longestPalindrome(self, words: List[str]) -> int:
        # 2131.Longest Palindrome by Concatenating Two Letter Words
        alphabet_size = 26
        count = [[0 for j in range(alphabet_size)] for i in range(alphabet_size)]
        for word in words:
            count[ord(word[0]) - ord("a")][ord(word[1]) - ord("a")] += 1
        answer = 0
        central = False
        for i in range(alphabet_size):
            if count[i][i] % 2 == 0:
                answer += count[i][i]
            else:
                answer += count[i][i] - 1
                central = True
            for j in range(i + 1, alphabet_size):
                answer += 2 * min(count[i][j], count[j][i])
        if central:
            answer += 1
        return 2 * answer

    def mostPoints(self, questions: List[List[int]]) -> int:
        # 2140.Solving Questions With Brainpower
        n = len(questions)
        dp = [0] * n

        def dfs(i):
            if i >= n:
                return 0
            if dp[i]:
                return dp[i]
            points, skip = questions[i]
            # dp[i] = max(skip it, solve it)
            dp[i] = max(dfs(i + 1), points + dfs(i + skip + 1))
            return dp[i]

        return dfs(0)

    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        # 2141.Maximum Running Time of N Computers
        left, right = 1, sum(batteries) // n
        while left < right:
            target = right - (right - left) // 2
            extra = 0
            for power in batteries:
                extra += min(power, target)
            if extra // n >= target:
                left = target
            else:
                right = target - 1
        return left

    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        # 2187.Minimum Time to Complete
        def cal_trips(time: List[int], now: int):
            trips = 0
            for t in time:
                trips += now // t
            return trips

        left = 0
        right = time[0] * totalTrips
        while left < right:
            mid = (left + right) // 2
            trips = cal_trips(time, mid)
            if trips < totalTrips:
                left = mid + 1
            else:
                right = mid
        return left
