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


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


def create_linked_list(nums: List[int]) -> Optional[ListNode]:
    if not nums:
        return None
    head = ListNode(nums[0])
    curr = head
    for num in nums[1:]:
        curr.next = ListNode(num)
        curr = curr.next
    return head


def create_binary_tree(nums: List[Optional[int]]) -> Optional[TreeNode]:
    def create_tree(index: int) -> Optional[TreeNode]:
        if index >= len(nums) or nums[index] is None:
            return None
        val = nums[index]
        if val is not None:
            node = TreeNode(val)
            node.left = create_tree(2 * index + 1)
            node.right = create_tree(2 * index + 2)
            return node
        return None

    return create_tree(0)
