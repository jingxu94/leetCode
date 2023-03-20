from typing import List


class Pro0601To0800:
    def __init__(self):
        pass

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

    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        # 744.Find Smallest Letter Greater Than Target
        if letters[-1] <= target:
            return letters[0]
        left, right = 0, len(letters) - 1
        while left < right:
            mid = (left + right) // 2
            if letters[mid] > target:
                right = mid
            elif letters[mid] <= target:
                left = mid + 1
        return letters[left]
