import heapq
import math
from queue import Queue
from typing import List


class Pro2401To2600:
    def __init__(self) -> None:
        pass

    def partitionString(self, s: str) -> int:
        # 2405.Optimal Partition of String
        last_seen = [-1] * 26
        count = 1
        starting = 0
        for i in range(len(s)):
            if last_seen[ord(s[i]) - ord("a")] >= starting:
                count += 1
                starting = i
            last_seen[ord(s[i]) - ord("a")] = i
        return count

    def minimizeArrayValue(self, nums: List[int]) -> int:
        # 2439.Minimize Maximum of Array
        ans = 0
        prefix_sum = 0
        for i in range(len(nums)):
            prefix_sum += nums[i]
            ans = max(ans, math.ceil(prefix_sum / (i + 1)))
        return ans

    def minCost(self, nums: List[int], cost: List[int]) -> int:
        # 2448.Minimum Cost to Make Array Equal
        def get_cost(base):
            return sum(abs(base - num) * c for num, c in zip(nums, cost))

        # Initialize the left and the right boundary of the binary search.
        left, right = min(nums), max(nums)
        answer = get_cost(nums[0])
        # As shown in the previous picture, if F(mid) > F(mid + 1), then the minimum
        # is to the right of mid, otherwise, the minimum is to the left of mid.
        while left < right:
            mid = (left + right) // 2
            cost_1 = get_cost(mid)
            cost_2 = get_cost(mid + 1)
            answer = min(cost_1, cost_2)
            if cost_1 > cost_2:
                left = mid + 1
            else:
                right = mid
        return answer

    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        # 2462.Total Cost to Hire K Workers
        # Add the first k workers with section id of 0 and
        # the last k workers with section id of 1 (without duplication) to pq.
        pq = []
        for i in range(candidates):
            pq.append((costs[i], 0))
        for i in range(max(candidates, len(costs) - candidates), len(costs)):
            pq.append((costs[i], 1))
        heapq.heapify(pq)
        answer = 0
        next_head, next_tail = candidates, len(costs) - 1 - candidates
        # Only refill pq if there are workers outside.
        for _ in range(k):
            cur_cost, cur_section_id = heapq.heappop(pq)
            answer += cur_cost
            if next_head <= next_tail:
                if cur_section_id == 0:
                    heapq.heappush(pq, (costs[next_head], 0))
                    next_head += 1
                else:
                    heapq.heappush(pq, (costs[next_tail], 1))
                    next_tail -= 1
        return answer

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        # 2466.Count Ways To Build Good Strings
        dp = [1] + [-1] * (high)
        mod = int(1e9 + 7)

        def dfs(end):  # pragma: no cover
            if dp[end] != -1:
                return dp[end]
            ans = 0
            if end >= zero:
                ans += dfs(end - zero)
            if end >= one:
                ans += dfs(end - one)
            dp[end] = ans % mod
            return dp[end]

        return sum(dfs(end) for end in range(low, high + 1)) % mod

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        # 2492.Minimum Score of a Path Between Two Cities
        ans = math.inf
        gr: List[List[tuple]] = [[] for _ in range(n + 1)]
        for edge in roads:
            gr[edge[0]].append((edge[1], edge[2]))  # u-> {v, dis}
            gr[edge[1]].append((edge[0], edge[2]))  # v-> {u, dis}
        vis = [0] * (n + 1)
        q: Queue = Queue()
        q.put(1)
        vis[1] = 1
        while not q.empty():
            node = q.get()
            for v, dis in gr[node]:
                ans = min(ans, dis)
                if vis[v] == 0:
                    vis[v] = 1
                    q.put(v)
        return int(ans)

    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        # 2542.Maximum Subsequence Score
        # Sort pair (nums1[i], nums2[i]) by nums2[i] in decreasing order.
        pairs = [(a, b) for a, b in zip(nums1, nums2)]
        pairs.sort(key=lambda x: -x[1])
        # Use a min-heap to maintain the top k elements.
        top_k_heap = [x[0] for x in pairs[:k]]
        top_k_sum = sum(top_k_heap)
        heapq.heapify(top_k_heap)
        # The score of the first k pairs.
        answer = top_k_sum * pairs[k - 1][1]
        # Iterate over every nums2[i] as minimum from nums2.
        for i in range(k, len(nums1)):
            # Remove the smallest integer from the previous top k elements
            # then ddd nums1[i] to the top k elements.
            top_k_sum -= heapq.heappop(top_k_heap)
            top_k_sum += pairs[i][0]
            heapq.heappush(top_k_heap, pairs[i][0])
            # Update answer as the maximum score.
            answer = max(answer, top_k_sum * pairs[i][1])
        return answer

    def splitNum(self, num: int) -> int:
        # 2578.Split With Minimum Sum
        digits = list(int(n) for n in str(num))
        digits.sort()
        digits.reverse()
        power1, power2 = len(digits) // 2, len(digits) - len(digits) // 2
        ans = 0
        for _ in range(power1):
            ans += 10 ** (power2 - 1) * digits[-1]
            digits.pop()
            ans += 10 ** (power1 - 1) * digits[-1]
            digits.pop()
            power1, power2 = power1 - 1, power2 - 1
        if power2:
            ans += digits[0]
        return ans
