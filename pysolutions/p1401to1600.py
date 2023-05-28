from collections import defaultdict
from typing import List
from .utils import TreeNode


class Pro1401To1600:
    def __init__(self):
        pass

    def stoneGameIII(self, stoneValue: List[int]) -> str:
        # 1406.Stone Game III
        n = len(stoneValue)
        dp = [0] * 4
        for i in range(n - 1, -1, -1):
            dp[i % 4] = max(sum(stoneValue[i : i + k]) - dp[(i + k) % 4] for k in range(1, 4))
        return "Alice" if dp[0] > 0 else "Bob" if dp[0] < 0 else "Tie"

    def numberOfArrays(self, s: str, k: int) -> int:
        # 1416.Restore The Array
        m, n = len(s), len(str(k))
        mod = 10**9 + 7
        # dp[i % (n + 1)] records the number of arrays that can be printed as
        # the prefix substring s[0 ~ i - 1]
        dp = [1] + [0] * n
        # Iterate over every digit, for each digit s[start]:
        for start in range(m):
            if s[start] == "0":
                dp[start % (n + 1)] = 0
                continue
            # Iterate over ending digit end and find all valid numbers
            # s[start ~ end].
            for end in range(start, m):
                if int(s[start : end + 1]) > k:
                    break
                # If s[start ~ end] is valid, increment dp[(end + 1) % (n + 1)] by dp[start].
                dp[(end + 1) % (n + 1)] += dp[start % (n + 1)]
                dp[(end + 1) % (n + 1)] %= mod
            # Set dp[start % (n + 1)] as 0.
            dp[start % (n + 1)] = 0
        return dp[m % (n + 1)]

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        # 1431.Kids With the Greatest Number of Candies
        max_candies = max(candies)
        return [candy + extraCandies >= max_candies for candy in candies]

    def ways(self, pizza: List[str], k: int) -> int:
        # 1444.Number of Ways of Cutting a Pizza
        rows = len(pizza)
        cols = len(pizza[0])
        apples = [[0] * (cols + 1) for _ in range(rows + 1)]
        for row in range(rows - 1, -1, -1):
            for col in range(cols - 1, -1, -1):
                apples[row][col] = (
                    (pizza[row][col] == "A") + apples[row + 1][col] + apples[row][col + 1] - apples[row + 1][col + 1]
                )
        dp = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(k)]
        dp[0] = [[int(apples[row][col] > 0) for col in range(cols)] for row in range(rows)]
        mod = 1000000007
        for remain in range(1, k):
            for row in range(rows):
                for col in range(cols):
                    val = 0
                    for next_row in range(row + 1, rows):
                        if apples[row][col] - apples[next_row][col] > 0:
                            val += dp[remain - 1][next_row][col]
                    for next_col in range(col + 1, cols):
                        if apples[row][col] - apples[row][next_col] > 0:
                            val += dp[remain - 1][row][next_col]
                    dp[remain][row][col] = val % mod
        return dp[k - 1][0][0]

    def goodNodes(self, root: TreeNode) -> int:
        # 1448.Count Good Nodes in Binary Tree
        def dfs(node, max_val):
            if not node:
                return 0
            ans = 0
            if node.val >= max_val:
                ans += 1
            ans += dfs(node.left, max(max_val, node.val))
            ans += dfs(node.right, max(max_val, node.val))
            return ans

        return dfs(root, root.val)

    def maxVowels(self, s: str, k: int) -> int:
        # 1456.Maximum Number of Vowels in a Substring of Given Length
        vowels = set("aeiou")
        max_vowels = 0
        cur_vowels = 0
        for i in range(k):
            if s[i] in vowels:
                cur_vowels += 1
        max_vowels = cur_vowels
        for i in range(k, len(s)):
            if s[i] in vowels:
                cur_vowels += 1
            if s[i - k] in vowels:
                cur_vowels -= 1
            max_vowels = max(max_vowels, cur_vowels)
        return max_vowels

    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        # 1466.Reorder Routes to Make All Paths Lead to the City Zero
        def dfs(node):
            checked.add(node)
            ans = 0
            for neighbor, need_reversed in roads[node]:
                if neighbor not in checked:
                    ans += need_reversed
                    ans += dfs(neighbor)
            return ans

        roads = defaultdict(list)
        for a, b in connections:
            roads[a].append((b, 1))  # Edge from a to b needs to be reversed
            roads[b].append((a, 0))  # Edge from b to a is already in the correct direction
        checked: set[int] = set()
        return dfs(0)

    def runningSum(self, nums: List[int]) -> List[int]:
        # 1480.Running Sum of 1d Array
        ans = nums
        if len(nums) > 1:
            for i in range(1, len(nums)):
                ans[i] = ans[i - 1] + nums[i]
        return ans

    def average(self, salary: List[int]) -> float:
        # 1491.Average Salary Excluding the Minimum and Maximum Salary
        return (sum(salary) - max(salary) - min(salary)) / (len(salary) - 2)

    def longestSubarray(self, nums: List[int]) -> int:
        # 1493.Longest Subarray of 1's After Deleting One Element
        ans = left = right = zeros = 0
        while right < len(nums):
            if nums[right] == 0:
                zeros += 1
            while zeros > 1:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            ans = max(ans, right - left)
            right += 1
        return ans

    def numSubseq(self, nums: List[int], target: int) -> int:
        # 1498.Number of Subsequences That Satisfy the Given Sum Condition
        nums.sort()
        mod = 1000000007
        ans = 0
        left = 0
        right = len(nums) - 1
        while left <= right:
            if nums[left] + nums[right] > target:
                right -= 1
            else:
                ans += pow(2, right - left, mod)
                left += 1
        return ans % mod

    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        # 1502.Can Make Arithmetic Progression From Sequence
        length = len(arr)
        if length == 2:
            return True
        arr.sort()
        gap = arr[0] - arr[1]
        for i in range(1, length - 1):
            if arr[i] - arr[i + 1] != gap:
                return False
        return True

    def countOdds(self, low: int, high: int) -> int:
        # 1523.Count Odd Numbers in an Interval Range
        nodd = (high - low) // 2
        if low % 2 or high % 2:
            nodd += 1
        return nodd

    def findKthPositive(self, arr: List[int], k: int) -> int:
        # 1539.Kth Missing Positive Number
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] - (mid + 1) < k:
                left = mid + 1
            else:
                right = mid
        return left + k

    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        # 1546.Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
        seen = set([0])
        ans = curr = 0
        for num in nums:
            curr += num
            prev = curr - target
            if prev in seen:
                ans += 1
                seen = set()
            seen.add(curr)
        return ans

    def minCost(self, n: int, cuts: List[int]) -> int:
        # 1547.Minimum Cost to Cut a Stick
        cuts.sort()
        cuts = [0] + cuts + [n]
        m = len(cuts)
        dp = [[0] * m for _ in range(m)]
        for i in range(m - 2, -1, -1):
            for j in range(i + 2, m):
                dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i + 1, j)) + cuts[j] - cuts[i]
        return dp[0][m - 1]

    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        # 1557.Minimum Number of Vertices to Reach All Nodes
        in_degrees = [0] * n
        for _, to in edges:
            in_degrees[to] += 1
        return [i for i in range(n) if in_degrees[i] == 0]

    def getMaxLen(self, nums: List[int]) -> int:
        # 1567.Maximum Length of Subarray With Positive Product
        ans = pos = neg = 0
        for x in nums:
            if x > 0:
                pos, neg = 1 + pos, 1 + neg if neg else 0
            elif x < 0:
                pos, neg = 1 + neg if neg else 0, 1 + pos
            else:
                pos = neg = 0
            ans = max(ans, pos)
        return ans

    def diagonalSum(self, mat: List[List[int]]) -> int:
        # 1572.Matrix Diagonal Sum
        rcs = len(mat)
        pdiag, sdiag = 0, 0
        for i in range(rcs):
            pdiag += mat[i][i]
            sdiag += mat[i][rcs - 1 - i]
        if rcs % 2 == 1:
            mid = rcs // 2
            sdiag -= mat[mid][mid]
        return pdiag + sdiag

    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # 1579.Remove Max Number of Edges to Keep Graph Fully Traversable
        def find(i, root):
            if i != root[i]:
                root[i] = find(root[i], root)
            return root[i]

        def union(i, j, root, size):
            root_i = find(i, root)
            root_j = find(j, root)
            if root_i != root_j:
                if size[root_i] < size[root_j]:
                    root_i, root_j = root_j, root_i
                root[root_j] = root_i
                size[root_i] += size[root_j]
                return True
            return False

        root_alice = [i for i in range(n + 1)]
        root_bob = [i for i in range(n + 1)]
        size_alice = [1] * (n + 1)
        size_bob = [1] * (n + 1)
        ans = 0
        for t, i, j in edges:
            if t == 3:
                if not union(i, j, root_alice, size_alice):
                    ans += 1
                else:
                    union(i, j, root_bob, size_bob)
        for t, i, j in edges:
            if t == 1:
                if not union(i, j, root_alice, size_alice):
                    ans += 1
            elif t == 2:
                if not union(i, j, root_bob, size_bob):
                    ans += 1
        if size_alice[find(1, root_alice)] != n or size_bob[find(1, root_bob)] != n:
            return -1
        return ans

    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        # 1588.Sum of All Odd Length Subarrays
        total_sum = 0
        n = len(arr)
        for i in range(n):
            left_count = i
            right_count = n - i - 1
            left_odd = (left_count + 1) // 2
            right_odd = (right_count + 1) // 2
            left_even = left_count // 2 + 1
            right_even = right_count // 2 + 1
            total_sum += arr[i] * (left_odd * right_odd + left_even * right_even)
        return total_sum


class BrowserHistory:  # pragma: no cover
    # 1472.Design Browser History
    def __init__(self, homepage: str):
        self._history: List[str] = []
        self._future: List[str] = []
        self._curr = homepage

    def visit(self, url: str) -> None:
        self._history.append(self._curr)
        self._curr = url
        self._future = []

    def back(self, steps: int) -> str:
        while steps > 0 and self._history:
            self._future.append(self._curr)
            self._curr = self._history.pop()
            steps -= 1
        return self._curr

    def forward(self, steps: int) -> str:
        while steps > 0 and self._future:
            self._history.append(self._curr)
            self._curr = self._future.pop()
            steps -= 1
        return self._curr
