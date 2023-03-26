import random
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional

from .utils import ListNode, TreeNode


class Pro0201To0400:
    def __init__(self):
        pass

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

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # 226.Invert Binary Tree
        if root is None:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

    def isPowerOfTwo(self, n: int) -> bool:
        # 231.Power of Two
        if n <= 0:
            return False
        while n % 2 == 0:
            n = n // 2
        return n == 1

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 206.Reverse Linked List
        curr, ans = head, None
        while curr:
            ans = ListNode(val=curr.val, next=ans)
            curr = curr.next
        return ans

    def containsDuplicate(self, nums: List[int]) -> bool:
        # 217.Contains Duplicate
        cont = Counter(nums)
        for key in cont.elements():
            if cont[key] > 1:
                return True
        return False

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

    def isAnagram(self, s: str, t: str) -> bool:
        # 242.Valid Anagram
        return Counter(s) == Counter(t)

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

    def reverseString(self, s: List[str]) -> List[str]:
        """344.Reverse String
        Do not return anything, modify s in-place instead.
        """
        s.reverse()
        return s

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
