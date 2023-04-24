from bisect import bisect_left
from collections import defaultdict, deque
from string import ascii_lowercase
from typing import Dict, List, Optional

from .utils import ListNode, TreeNode


class Pro1201To1400:
    def __init__(self):
        pass

    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        """1232.Check If It Is a Straight Line
        Check if the given coordinates form a straight line.

        Args:
        coordinates (List[List[int]]): List of x, y coordinates.

        Returns:
        bool: True if coordinates form a straight line, False otherwise.
        """
        if len(coordinates) < 3:
            return True
        dx, dy = coordinates[0][0] - coordinates[1][0], coordinates[0][1] - coordinates[1][1]
        for point in coordinates[2:]:
            if (coordinates[0][0] - point[0]) * dy != (coordinates[0][1] - point[1]) * dx:
                return False
        return True

    def minRemoveToMakeValid(self, s: str) -> str:
        # 1249.Minimum Remove to Make Valid Parentheses
        stack: List[int] = []
        for i, c in enumerate(s):
            if c == "(":
                stack.append(i)
            elif c == ")":
                if stack:
                    stack.pop()
                else:
                    s = s[:i] + "*" + s[i + 1 :]
        for i in stack:
            s = s[:i] + "*" + s[i + 1 :]
        return s.replace("*", "")

    def closedIsland(self, grid: List[List[int]]) -> int:
        # 1254.Number of Closed Islands
        m, n = len(grid), len(grid[0])

        def dfs(grid, row, col):
            if row < 0 or row >= m or col < 0 or col >= n or grid[row][col] != 0:
                return
            grid[row][col] = 1
            dfs(grid, row - 1, col)
            dfs(grid, row + 1, col)
            dfs(grid, row, col - 1)
            dfs(grid, row, col + 1)

        for row in range(m):
            for col in range(n):
                if (row * col == 0 or row == m - 1 or col == n - 1) and grid[row][col] == 0:
                    dfs(grid, row, col)
        num_closed_islands = 0
        for row in range(1, m - 1):
            for col in range(1, n - 1):
                if grid[row][col] == 0:
                    num_closed_islands += 1
                    dfs(grid, row, col)
        return num_closed_islands

    def subtractProductAndSum(self, n: int) -> int:
        """1281.Subtract the Product and Sum of Digits of an Integer
        Subtract the product of digits and the sum of digits of an integer.

        Args:
        n (int): The integer.

        Returns:
        int: The difference between the product and sum of the digits of the integer.
        """
        digits = str(n)
        pd, sm = 1, 0
        for i in range(len(digits)):
            pd *= int(digits[i])
            sm += int(digits[i])
        return pd - sm

    def getDecimalValue(self, head: ListNode) -> int:
        # 1290.Convert Binary Number in a Linked List to Integer
        curr, ans = head, 0
        while curr:
            ans = 2 * ans + curr.val
            curr = curr.next
        return ans

    def freqAlphabets(self, s: str) -> str:
        # 1309.Decrypt String from Alphabet to Integer Mapping
        ans = ""
        if "#" not in s:
            for num in s:
                ans += ascii_lowercase[int(num) - 1]
            return ans
        nums_list = s.split("#")
        for nums in nums_list[:-1]:
            for num in nums[:-2]:
                ans += ascii_lowercase[int(num) - 1]
            ans += ascii_lowercase[int(nums[-2:]) - 1]
        for num in nums_list[-1]:
            ans += ascii_lowercase[int(num) - 1]
        return ans

    def minInsertions(self, s: str) -> int:
        # 1312.Minimum Insertion Steps to Make a String Palindrome
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                dp[i][j] = dp[i + 1][j - 1] if s[i] == s[j] else min(dp[i + 1][j], dp[i][j - 1]) + 1
        return dp[0][n - 1]

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        # 1314.Matrix Block Sum
        m, n = len(mat), len(mat[0])
        # Calculate the prefix sum matrix
        prefix_sum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                prefix_sum[i + 1][j + 1] = mat[i][j] + prefix_sum[i + 1][j] + prefix_sum[i][j + 1] - prefix_sum[i][j]
        # Calculate the block sums
        result = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                r1, c1 = max(0, i - k), max(0, j - k)
                r2, c2 = min(m, i + k + 1), min(n, j + k + 1)
                result[i][j] = prefix_sum[r2][c2] - prefix_sum[r1][c2] - prefix_sum[r2][c1] + prefix_sum[r1][c1]
        return result

    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        # 1319.Number of Operations to Make Network Connected
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            root_x = find(x)
            root_y = find(y)
            # root_x != root_y:
            parent[root_x] = root_y

        # Initialize the parent array
        parent = [i for i in range(n)]
        # Count redundant connections
        redundant = 0
        for x, y in connections:
            if find(x) == find(y):
                redundant += 1
            else:
                union(x, y)
        # Count the number of connected components
        components = sum([1 for i in range(n) if parent[i] == i])
        # Check if there are enough connections to form a connected
        if redundant < components - 1:
            return -1
        else:
            return components - 1

    def maximum69Number(self, num: int) -> int:
        """1323.Maximum 69 Number
        Change the first 6 in the given number to 9 and return the resulting number.

        Args:
        num (int): The input number.

        Returns:
        int: The maximum number that can be formed by changing the first 6 to 9.
        """
        numstr = str(num)
        if "6" not in numstr:
            return num
        return num + 3 * 10 ** (len(numstr) - numstr.index("6") - 1)

    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        """1337.The K Weakest Rows in a Matrix
        Find the K weakest rows in a matrix based on the number of soldiers.

        Args:
        mat (List[List[int]]): The input matrix.
        k (int): The number of weakest rows to return.

        Returns:
        List[int]: List of indices of the K weakest rows.
        """
        soldiers = []
        for row in range(len(mat)):
            soldiers.append(sum(mat[row]))
        arg_soldiers = sorted(range(len(soldiers)), key=soldiers.__getitem__)
        return arg_soldiers[:k]

    def numberOfSteps(self, num: int) -> int:
        """1342.Number of Steps to Reduce a Number to Zero
        Calculate the number of steps to reduce a number to zero by dividing by 2
        if it's even or subtracting 1 if it's odd.

        Args:
        num (int): The input number.

        Returns:
        int: The number of steps to reduce the number to zero.
        """
        steps = 0
        while num > 0:
            if num % 2 == 0:
                num = num // 2
            else:
                num -= 1
            steps += 1
        return steps

    def checkIfExist(self, arr: List[int]) -> bool:
        # 1346.Check If N and Its Double Exist
        seen = set()
        for num in arr:
            if num * 2 in seen or (num % 2 == 0 and num // 2 in seen):
                return True
            seen.add(num)
        return False

    def countNegatives(self, grid: List[List[int]]) -> int:
        # 1351.Count Negative Numbers in a Sorted Matrix
        def binary_search(row: List[int]) -> int:
            left, right = 0, len(row)
            while left < right:
                mid = (left + right) // 2
                if row[mid] < 0:
                    right = mid
                else:
                    left = mid + 1
            return len(row) - left

        count = 0
        for row in grid:
            count += binary_search(row)
        return count

    def sortByBits(self, arr: List[int]) -> List[int]:
        # 1356.Sort Integers by The Number of 1 Bits
        nbits = defaultdict(list)
        arr.sort()
        for i, num in enumerate(arr):
            num_ones = 0
            while num > 0:
                if num % 2 == 1:
                    num_ones += 1
                num = num // 2
            nbits[num_ones].append(arr[i])
        ans = []
        keys = list(nbits.keys())
        keys.sort()
        for key in keys:
            ans.extend(nbits[key])
        return ans

    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:  # pragma: no cover
        # 1367.Linked List in Binary Tree
        if not root:
            return False

        def check_path(head: Optional[ListNode], node: Optional[TreeNode]) -> bool:
            if not head:
                return True
            if not node or node.val != head.val:
                return False
            return check_path(head.next, node.left) or check_path(head.next, node.right)

        return check_path(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)

    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        # 1372.Longest ZigZag Path in a Binary Tree
        path_lenght = 0

        def dfs(node, goleft, steps):
            if node:
                nonlocal path_lenght
                path_lenght = max(path_lenght, steps)
                if goleft:
                    dfs(node.left, False, steps + 1)
                    dfs(node.right, True, 1)
                else:
                    dfs(node.left, False, 1)
                    dfs(node.right, True, steps + 1)

        dfs(root, False, 0)
        dfs(root, True, 0)
        return path_lenght

    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        # 1376.Time Needed to Inform All Employees
        hierarchy: Dict[int, List[int]] = {}
        for i, m in enumerate(manager):
            if m not in hierarchy:
                hierarchy[m] = []
            hierarchy[m].append(i)
        queue = deque([(headID, 0)])
        max_time = 0
        while queue:
            curr, time = queue.popleft()
            max_time = max(max_time, time)
            if curr in hierarchy:
                for emp in hierarchy[curr]:
                    queue.append((emp, time + informTime[curr]))
        return max_time

    def findTheDistanceValue1(self, arr1: List[int], arr2: List[int], d: int) -> int:
        """1385.Find the Distance Value Between Two Arrays
        Find the distance value between two arrays.

        Args:
        arr1 (List[int]): The first array.
        arr2 (List[int]): The second array.
        d (int): The distance value.

        Returns:
        int: The distance value between two arrays.
        """
        check: List[int] = []
        for i in range(-d, d + 1):
            check.extend(map(lambda x: x + i, arr2))
        ans = 0
        for num in arr1:
            if num not in check:
                ans += 1
        return ans

    def findTheDistanceValue2(self, arr1: List[int], arr2: List[int], d: int) -> int:
        """1385.Find the Distance Value Between Two Arrays
        Find the distance value between two arrays using bisect_left.

        Args:
        arr1 (List[int]): The first array.
        arr2 (List[int]): The second array.
        d (int): The distance value.

        Returns:
        int: The distance value between two arrays.
        """
        # Solution2: Using bisect_left
        arr2.sort()
        n = len(arr2)
        count = 0
        for x in arr1:
            i = bisect_left(arr2, x)
            if (i == n or arr2[i] - x > d) and (i == 0 or x - arr2[i - 1] > d):
                count += 1
        return count
