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
