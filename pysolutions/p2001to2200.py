from typing import List, Optional
from .utils import ListNode


class Pro2001To2200:
    def __init__(self):
        pass

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
