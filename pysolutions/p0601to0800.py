from collections import Counter
from typing import List, Optional

from .utils import TreeNode


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
                prod /= nums[left]
                left += 1
            ans += right - left + 1
        return ans

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
