from collections import deque
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def set_ListNode(numbers: List[int]) -> Optional[ListNode]:
    result = ListNode()
    curr = result
    for num in numbers:
        curr.next = ListNode(num)
        curr = curr.next
    return result.next


def set_TreeNode(numbers: List[int]) -> Optional[TreeNode]:
    nnode = len(numbers)
    if nnode == 0:
        return None
    root = TreeNode()
    tree_queue = deque([root])
    for i in range(len(numbers)):
        node = tree_queue.popleft()
        node.val = numbers[i]
        nnode -= 1
        if nnode:
            node.left = TreeNode()
            nnode -= 1
            tree_queue.extend([node.left])
        if nnode:
            node.right = TreeNode()
            nnode -= 1
            tree_queue.extend([node.right])
    return root
