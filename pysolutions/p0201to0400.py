import heapq
import random
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Set

from .utils import ListNode, TreeNode


class Pro0201To0400:
    def __init__(self):
        pass

    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        # 201.Bitwise AND of Numbers Range
        shift = 0
        while left < right:
            left >>= 1
            right >>= 1
            shift += 1
        return left << shift

    def isHappy(self, n: int) -> bool:
        # 202.Happy Number
        checked = []
        while n != 1 and n not in checked:
            checked.append(n)
            num = 0
            while n // 10 > 0:
                num += (n % 10) ** 2
                n = n // 10
            num += (n % 10) ** 2
            n = num
        if n == 1:
            return True
        return False

    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        # 203.Remove Linked List Elements
        foward = ListNode(0, next=head)
        ans = ListNode()
        curr = ans
        while foward.next:
            foward = foward.next
            if foward.val != val:
                curr.next = ListNode(foward.val)
                curr = curr.next
        return ans.next

    def isIsomorphic(self, s: str, t: str) -> bool:
        # 205.Isomoriphic Strings
        if len(s) != len(t):
            return False
        mapping_st: Dict[str, str] = dict()
        mapping_ts: Dict[str, str] = dict()
        for i in range(len(s)):
            if mapping_ts.get(t[i]) is None:
                mapping_ts[t[i]] = s[i]
            if mapping_st.get(s[i]) is None:
                mapping_st[s[i]] = t[i]
            if s[i] != mapping_ts[t[i]] or t[i] != mapping_st[s[i]]:
                return False
        return True

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 206.Reverse Linked List
        curr, ans = head, None
        while curr:
            ans = ListNode(val=curr.val, next=ans)
            curr = curr.next
        return ans

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 207.Course Schedule
        adj_list: defaultdict = defaultdict(list)
        indegree: Dict[int, int] = {}
        for dest, src in prerequisites:
            adj_list[src].append(dest)
            indegree[dest] = indegree.get(dest, 0) + 1
        zero_indegree_queue = deque([k for k in range(numCourses) if k not in indegree])
        topological_sorted_order: List[int] = []
        while zero_indegree_queue:
            vertex = zero_indegree_queue.popleft()
            topological_sorted_order.append(vertex)
            if vertex in adj_list:
                for neighbor in adj_list[vertex]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        zero_indegree_queue.append(neighbor)
        if len(topological_sorted_order) == numCourses:
            return True
        return False

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 209.Minimum Size Subarray Sum
        left = total = 0
        ans = None
        for right, num in enumerate(nums):
            total += num
            while total >= target:
                ans = min(ans or float("inf"), right - left + 1)
                total -= nums[left]
                left += 1
        if ans:
            return int(ans)
        else:
            return 0

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 210.Course Schedule II
        adj_list: defaultdict = defaultdict(list)
        indegree: Dict[int, int] = {}
        for dest, src in prerequisites:
            adj_list[src].append(dest)
            indegree[dest] = indegree.get(dest, 0) + 1
        zero_indegree_queue = deque([k for k in range(numCourses) if k not in indegree])
        topological_sorted_order: List[int] = []
        while zero_indegree_queue:
            vertex = zero_indegree_queue.popleft()
            topological_sorted_order.append(vertex)
            if vertex in adj_list:
                for neighbor in adj_list[vertex]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        zero_indegree_queue.append(neighbor)
        if len(topological_sorted_order) == numCourses:
            return topological_sorted_order
        return []

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # 212.Word Search II
        # Define a DFS function to traverse the board and search for words
        def dfs(x, y, root):
            # Get the letter at the current position on the board
            letter = board[x][y]
            # Traverse the trie to the next node
            cur = root[letter]
            # Check if the node has a word in it
            word = cur.pop("#", False)
            if word:
                # If a word is found, add it to the results list
                res.append(word)
            # Mark the current position on the board as visited
            board[x][y] = "*"
            # Recursively search in all four directions
            for dirx, diry in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                curx, cury = x + dirx, y + diry
                # Check if the next position is within the board and the next letter is in the trie
                if 0 <= curx < m and 0 <= cury < n and board[curx][cury] in cur:
                    dfs(curx, cury, cur)
            # Restore the original value of the current position on the board
            board[x][y] = letter
            # If the current node has no children, remove it from the trie
            if not cur:
                root.pop(letter)

        # Build a trie data structure from the list of words
        trie: dict = {}
        for word in words:
            cur = trie
            for letter in word:
                cur = cur.setdefault(letter, {})
            cur["#"] = word
        # Get the dimensions of the board
        m, n = len(board), len(board[0])
        # Initialize a list to store the results
        res: List[str] = []
        # Traverse the board and search for words
        for i in range(m):
            for j in range(n):
                # Check if the current letter is in the trie
                if board[i][j] in trie:
                    dfs(i, j, trie)
        # Return the list of results
        return res

    def rob(self, nums: List[int]) -> int:
        # 213.House Robber II
        def rob_helper(nums: List[int]) -> int:
            prev, curr = 0, 0
            for num in nums:
                prev, curr = curr, max(prev + num, curr)
            return curr

        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        return max(rob_helper(nums[:-1]), rob_helper(nums[1:]))

    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 215.Kth Largest Element in an Array
        return heapq.nlargest(k, nums)[-1]

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        # 216.Combination Sum III
        ans: List[List[int]] = []

        def dfs(start: int, path: List[int]):
            if len(path) == k and sum(path) == n:
                ans.append(path)
                return
            for i in range(start, 10):
                dfs(i + 1, path + [i])

        dfs(1, [])
        return ans

    def containsDuplicate(self, nums: List[int]) -> bool:
        # 217.Contains Duplicate
        cont = Counter(nums)
        for key in cont.elements():
            if cont[key] > 1:
                return True
        return False

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        # 221.Maximal Square
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ans = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i - 1][j - 1] == "1":
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
                    ans = max(ans, dp[i][j])
        return ans**2

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # 226.Invert Binary Tree
        if root is None:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

    def summaryRanges(self, nums: List[int]) -> List[str]:
        # 228.Summary Ranges
        ans: List[str] = []
        i = 0
        while i < len(nums):
            low = i
            i += 1
            while i < len(nums) and nums[i] == nums[i - 1] + 1:
                i += 1
            high = i - 1
            if low < high:
                ans.append(str(nums[low]) + "->" + str(nums[high]))
            else:
                ans.append(str(nums[low]))
        return ans

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # 230.Kth Smallest Element in a BST
        def inorder(root: Optional[TreeNode]) -> List[int]:
            if root is None:
                return []
            return inorder(root.left) + [root.val] + inorder(root.right)

        return inorder(root)[k - 1]

    def isPowerOfTwo(self, n: int) -> bool:
        # 231.Power of Two
        if n <= 0:
            return False
        while n % 2 == 0:
            n = n // 2
        return n == 1

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # 234.Palindrome Linked List
        if head is None:
            return False
        fd_mid, fd_end = head, head
        tmp_reverse = None
        while fd_end and fd_end.next:
            fd_mid = fd_mid.next
            fd_end = fd_end.next.next
        tmp_reverse, fd_mid, tmp_reverse.next = fd_mid, fd_mid.next, None
        while fd_mid:
            fd_mid.next, tmp_reverse, fd_mid = tmp_reverse, fd_mid, fd_mid.next
        fd_end, fd_mid = head, tmp_reverse
        while fd_mid:
            if fd_end.val != fd_mid.val:
                return False
            fd_end, fd_mid = fd_end.next, fd_mid.next
        return True

    def lowestCommonAncestor(self, root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]):
        # 235.Lowest Common Ancestor of a Binary Search Tree
        if p is None:
            return q
        if q is None:
            return p
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
        else:
            return None

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # 238.Product of Array Except Self
        tprod = 1
        for num in nums:
            tprod *= num
        if tprod != 0:
            return list(tprod // num for num in nums)
        ct = Counter(nums)
        ans = [0] * len(nums)
        if ct[0] > 1:
            return ans
        else:
            index = nums.index(0)
            tprod = 1
            for i, num in enumerate(nums):
                if i != index:
                    tprod *= num
            ans[index] = tprod
            return ans

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 239.Sliding Window Maximum
        if not nums:
            return []
        # Initialize deque and result list
        window: deque = deque()
        result = []

        for i, num in enumerate(nums):
            # Remove elements outside the current window from the front of the deque
            while window and window[0] < i - k + 1:
                window.popleft()
            # Remove elements smaller than the current element from the back of the deque
            while window and nums[window[-1]] < num:
                window.pop()
            # Add the current index to the back of the deque
            window.append(i)
            # Add the maximum value (front of the deque) to the result list
            if i >= k - 1:
                result.append(nums[window[0]])
        return result

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 240.Search a 2D Matrix II
        rows, cols = len(matrix), len(matrix[0])
        row, col = 0, cols - 1
        while row < rows and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False

    def isAnagram(self, s: str, t: str) -> bool:
        # 242.Valid Anagram
        return Counter(s) == Counter(t)

    def addDigits(self, num: int) -> int:
        # 258.Add Digits
        if num == 0:
            return 0
        return (num - 1) % 9 + 1

    def nthUglyNumber(self, n: int) -> int:
        # 264.Ugly Number II
        ugly_numbers = [1]
        seen = {1}
        factors = [2, 3, 5]
        heap = [(f, 0, f) for f in factors]
        heapq.heapify(heap)
        while len(ugly_numbers) < n:
            val, idx, factor = heapq.heappop(heap)
            if val not in seen:
                seen.add(val)
                ugly_numbers.append(val)
            heapq.heappush(heap, (factor * ugly_numbers[idx + 1], idx + 1, factor))
        return ugly_numbers[-1]

    def firstBadVersion(self, n: int) -> int:  # pragma: no cover
        # 278.First Bad Version
        def isBadVersion(version: int) -> bool:
            return random.choice([True, False])

        left, right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left

    def numSquares(self, n: int) -> int:
        # 279.Perfect Squares
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = min(dp[i - j * j] for j in range(1, int(i**0.5) + 1)) + 1
        return dp[-1]

    def moveZeroes(self, nums: List[int]) -> List[int]:
        """283.Move Zeroes
        Do not return anything, modify nums in-place instead.
        """
        nactivate = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[nactivate] = nums[i]
                nactivate += 1
        for i in range(nactivate, len(nums)):
            nums[i] = 0

        return nums

    def findDuplicate(self, nums: List[int]) -> int:
        # 287.Find the Duplicate Number
        slow, fast = nums[0], nums[nums[0]]
        while slow != fast:
            slow, fast = nums[slow], nums[nums[fast]]
        slow = 0
        while slow != fast:
            slow, fast = nums[slow], nums[fast]
        return slow

    def wordPattern(self, pattern: str, s: str) -> bool:
        # 290.Word Pattern
        dict_f: defaultdict[str, str] = defaultdict()
        s_list = s.split(" ")
        if len(s_list) != len(pattern):
            return False
        for i, word in enumerate(s_list):
            word = word + "#"
            if pattern[i] not in dict_f.keys() and word not in dict_f.keys():
                dict_f[pattern[i]] = word
                dict_f[word] = pattern[i]
            elif pattern[i] not in dict_f.keys() or word not in dict_f.keys():
                return False
            elif dict_f[pattern[i]] != word or dict_f[word] != pattern[i]:
                return False
        return True

    def getHint(self, secret: str, guess: str) -> str:
        # 299.Bulls and Cows
        newsct = newgus = ""
        bulls = cows = 0
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                bulls += 1
            else:
                newsct += secret[i]
                newgus += guess[i]
        cts = Counter(newsct)
        ctg = Counter(newgus)
        for key in ctg.keys():
            if key in cts.keys():
                cows += min(ctg[key], cts[key])
        return "".join([str(bulls), "A", str(cows), "B"])

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 300.Longest Increasing Subsequence
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def maxProfit(self, prices: List[int]) -> int:
        # 309.Best Time to Buy and Sell Stock with Cooldown
        if len(prices) == 1:
            return 0
        dp = [[0] * 2 for _ in range(len(prices))]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        dp[1][0] = max(0, prices[1] - prices[0])
        dp[1][1] = -min(prices[0], prices[1])
        for k in range(2, len(prices)):
            dp[k][1] = max(dp[k - 1][1], dp[k - 2][0] - prices[k])
            dp[k][0] = max(dp[k - 1][0], dp[k - 1][1] + prices[k])
        return dp[-1][0]

    def bulbSwitch(self, n: int) -> int:
        # 319.Bulb Switcher
        return int(n**0.5)

    def coinChange(self, coins: List[int], amount: int) -> int:
        # 322.Coin Change
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return int(dp[-1]) if dp[-1] != float("inf") else -1

    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 328.Odd Even Linked List
        if not head:
            return head
        odd = head
        even = head.next
        even_head = even
        while even and even.next:
            odd.next = even.next
            even.next = even.next.next
            even = even.next
            odd = odd.next
        odd.next = even_head
        return head

    def increasingTriplet(self, nums: List[int]) -> bool:
        # 334.Increasing Triplet Subsequence
        first = second = float("inf")
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True
        return False

    def countBits(self, n: int) -> List[int]:
        # 338.Counting Bits
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        return dp

    def integerBreak(self, n: int) -> int:
        # 343.Integer Break
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]
        # ================================
        # if n < 4:
        #     return n - 1
        # elif n % 3 == 0:
        #     return 3 ** (n // 3)
        # elif n % 3 == 1:
        #     return 3 ** (n // 3 - 1) * 4
        # else:
        #     return 3 ** (n // 3) * 2

    def reverseString(self, s: List[str]) -> List[str]:
        """344.Reverse String
        Do not return anything, modify s in-place instead.
        """
        s.reverse()
        return s

    def reverseVowels(self, s: str) -> str:
        # 345.Reverse Vowels of a String
        vowels = "aeiouAEIOU"
        s_list = list(s)
        left, right = 0, len(s_list) - 1
        while left < right:
            if s_list[left] not in vowels:
                left += 1
            elif s_list[right] not in vowels:
                right -= 1
            else:
                s_list[left], s_list[right] = s_list[right], s_list[left]
                left += 1
                right -= 1
        return "".join(s_list)

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 347.Top K Frequent Elements
        return [key for key, _ in Counter(nums).most_common(k)]

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 350.Intersection of Two Arrays 2
        cnums1 = Counter(nums1)
        cnums2 = Counter(nums2)
        for key in cnums1.keys():
            num = cnums2.get(key, 0)
            cnums1[key] = min(num, cnums1[key])
        return list(cnums1.elements())

    def isPerfectSquare(self, num: int) -> bool:
        # 367.Valid Perfect Square
        left, right = 1, num
        while left <= right:
            mid = (left + right) // 2
            square = mid**2
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        return False

    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # 373.Find K Pairs with Smallest Sums
        if not nums1 or not nums2:
            return []
        heap: List[tuple[int, List[int]]] = []
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                if len(heap) < k:
                    heapq.heappush(heap, (-nums1[i] - nums2[j], [nums1[i], nums2[j]]))
                else:
                    if -heap[0][0] > nums1[i] + nums2[j]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, (-nums1[i] - nums2[j], [nums1[i], nums2[j]]))
                    else:
                        break
        return [item[1] for item in heap]

    def guessNumber(self, n: int) -> int:  # pragma: no cover
        # 374.Guess Number Higher or Lower
        def guess(num: int) -> int:
            return random.choice([-1, 0, 1])

        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            callback = guess(mid)
            if callback == 0:
                return mid
            elif callback < 0:
                right = mid - 1
            else:
                left = mid + 1
        return -1

    def wiggleMaxLength(self, nums: List[int]) -> int:
        # 376.Wiggle Subsequence
        if len(nums) < 2:
            return len(nums)
        up = down = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                up = down + 1
            elif nums[i] < nums[i - 1]:
                down = up + 1
        return max(up, down)

    def combinationSum4(self, nums: List[int], target: int) -> int:
        # 377.Combination Sum IV
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        return dp[-1]

    def getRandom(self, head: Optional[ListNode]) -> int:  # pragma: no cover
        # 382.Linked List Random Node
        maxnode = 1
        result = 0
        curr = head
        while curr:
            if random.random() < 1 / maxnode:
                result = curr.val
            curr = curr.next
            maxnode += 1
        return result

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # 383.Ransom Note
        note, mag = Counter(ransomNote), Counter(magazine)
        return (note & mag) == note

    def firstUniqChar(self, s: str) -> int:
        # 387.First Unique Character in a String
        if s == "":
            return -1
        count = Counter(s)
        # if all(count[key] > 1 for key in count.keys()):
        #     return -1
        for i in range(len(s)):
            if count[s[i]] == 1:
                return i
        return -1

    def findTheDifference(self, s: str, t: str) -> str:
        # 389.Find the Difference
        counts, countt = Counter(s), Counter(t)
        for key in countt.keys():
            if countt[key] - counts.get(key, 0) != 0:
                return key
        return ""

    def isSubsequence(self, s: str, t: str) -> bool:
        # 392.Is Subsequence
        if s == "":
            return True
        for i in range(len(t)):
            if s[0] == t[i]:
                return self.isSubsequence(s[1:], t[i + 1 :])
        return False

    def decodeString(self, s: str) -> str:
        # 394.Decode String
        # S1: Using re
        # while '[' in s:
        #     s = re.sub(r'(\d+)\[([a-zA-Z]*)\]', lambda m: int(m.group(1)) * m.group(2), s)
        # return s
        # ========================
        # S2: Using stack
        stack: List[str] = []
        for ch in s:
            if ch == "]":
                repeat_str = ""
                while stack and stack[-1] != "[":
                    repeat_str = stack.pop() + repeat_str
                stack.pop()  # remove '[' from stack
                # Get the number of repetitions
                k = ""
                while stack and stack[-1].isdigit():
                    k = stack.pop() + k
                nk = int(k)
                stack.append(repeat_str * nk)
            else:
                stack.append(ch)
        return "".join(stack)

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 399.Evaluate Division
        graph: defaultdict = defaultdict(dict)
        for (x, y), v in zip(equations, values):
            graph[x][y] = v
            graph[y][x] = 1 / v

        def dfs(x: str, y: str, visited: Set[str]) -> float:  # pragma: no cover
            if y in graph[x]:
                return graph[x][y]
            for z in graph[x]:
                if z not in visited:
                    visited.add(z)
                    d = dfs(z, y, visited)
                    if d > 0:
                        return graph[x][z] * d
            return -1.0

        return [dfs(x, y, set()) if x in graph and y in graph else -1.0 for x, y in queries]


class Trie:  # pragma: no cover
    # 208.Implement Trie (Prefix Tree)
    def __init__(self):
        self.words = []

    def insert(self, word: str) -> None:
        self.words.append(word)

    def search(self, word: str) -> bool:
        if word in self.words:
            return True
        return False

    def startsWith(self, prefix: str) -> bool:
        for word in self.words:
            if word.startswith(prefix):
                return True
        return False


class WordDictionaryS1:  # pragma: no cover
    # 211.Design Add and Search Words Data Structure
    def __init__(self):
        self.words = defaultdict(list)

    def addWord(self, word: str) -> None:
        self.words[len(word)].append(word)

    def search(self, word: str) -> bool:
        n = len(word)
        if "." in word:
            for w in self.words[n]:
                if all(word[i] in (w[i], ".") for i in range(n)):
                    return True
            else:
                return False
        return word in self.words[n]


class WordDictionaryS2:  # pragma: no cover
    # 211.Design Add and Search Words Data Structure
    def __init__(self):
        self.wdict = dict()

    def addWord(self, word: str) -> None:
        nword_list = self.wdict.get(str(len(word)), [])
        if word not in nword_list:
            nword_list.append(word)
            self.wdict[str(len(word))] = nword_list

    def search(self, word: str) -> bool:
        nword_list = self.wdict.get(str(len(word)), [])
        if nword_list == []:
            return False
        if "." in word:
            for nword in nword_list:
                if all(word[i] in [nword[i], "."] for i in range(len(word))):
                    return True
            return False
        return word in nword_list


class MyQueue:  # pragma: no cover
    # 232.Implement Queue using Stacks
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        self.stack1.append(x)

    def pop(self) -> int:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self) -> int:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self) -> bool:
        return not self.stack1 and not self.stack2


class Solution:  # pragma: no cover
    # 236.Lowest Common Ancestor of a Binary Tree
    def lowestCommonAncestor(self, root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left or right


class MedianFinder:  # pragma: no cover
    # 295.Find Median from Data Stream
    def __init__(self):
        self.nums = []

    def addNum(self, num: int) -> None:
        self.nums.append(num)

    def findMedian(self) -> float:
        self.nums.sort()
        n = len(self.nums)
        if n % 2 == 0:
            return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2
        else:
            return self.nums[n // 2]


# from sortedcontainers import SortedList
# class MedianFinder_v2: # pragma: no cover
#     # 295.Find Median from Data Stream
#     def __init__(self):
#         self.nums = SortedList([])

#     def addNum(self, num: int) -> None:
#         self.nums.add(num)

#     def findMedian(self) -> float:
#         n = len(self.nums)
#         if n % 2 == 0:
#             return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2
#         else:
#             return self.nums[n // 2]


class Codec:  # pragma: no cover
    # 297.Serialize and Deserialize Binary Tree
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string."""
        if not root:
            return ""
        queue = deque([root])
        res = []
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append("null")
        return ",".join(res)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree."""
        if not data:
            return None
        lsdata = data.split(",")
        root = TreeNode(int(lsdata[0]))
        queue = deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if lsdata[i] != "null":
                node.left = TreeNode(int(lsdata[i]))
                queue.append(node.left)
            i += 1
            if lsdata[i] != "null":
                node.right = TreeNode(int(lsdata[i]))
                queue.append(node.right)
            i += 1
        return root


class NumArray:  # pragma: no cover
    # 303.Range Sum Query - Immutable
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sums = [0]
        for num in nums:
            self.sums.append(self.sums[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        return self.sums[right + 1] - self.sums[left]


class NumMatrix:  # pragma: no cover
    # 304.Range Sum Query 2D - Immutable
    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            return
        m, n = len(matrix), len(matrix[0])
        self.sums = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.sums[i + 1][j + 1] = self.sums[i][j + 1] + self.sums[i + 1][j] - self.sums[i][j] + matrix[i][j]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return (
            self.sums[row2 + 1][col2 + 1]
            - self.sums[row2 + 1][col1]
            - self.sums[row1][col2 + 1]
            + self.sums[row1][col1]
        )


class NestedIterator:  # pragma: no cover
    # 341.Flatten Nested List Iterator
    def __init__(self, nestedList):
        self.stack = nestedList[::-1]

    def next(self):
        return self.stack.pop().getInteger()

    def hasNext(self):
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            self.stack = self.stack[:-1] + top.getList()[::-1]
        return False


class RandomizedSet:  # pragma: no cover
    # 380.Insert Delete GetRandom O(1)
    def __init__(self):
        self.nums = []
        self.pos = {}

    def insert(self, val: int) -> bool:
        if val not in self.pos:
            self.nums.append(val)
            self.pos[val] = len(self.nums) - 1
            return True
        return False

    def remove(self, val: int) -> bool:
        if val in self.pos:
            last, idx = self.nums[-1], self.pos[val]
            self.nums[idx], self.pos[last] = last, idx
            self.nums.pop()
            self.pos.pop(val)
            return True
        return False

    def getRandom(self) -> int:
        return random.choice(self.nums)


class Solution384:  # pragma: no cover
    # 384.Shuffle an Array
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.original = list(nums)

    def reset(self) -> List[int]:
        self.nums = self.original
        self.original = list(self.original)
        return self.nums

    def shuffle(self) -> List[int]:
        random.shuffle(self.nums)
        return self.nums
