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
    def __init__(self, val: int = 0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


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
        if index >= len(nums):
            return None
        val = nums[index]
        if val is not None:
            node = TreeNode(val)
            node.left = create_tree(2 * index + 1)
            node.right = create_tree(2 * index + 2)
            return node
        else:
            return None

    return create_tree(0)

    # def create_tree() -> Optional[TreeNode]:
    #     nonlocal index
    #     if index >= len(nums) or nums[index] is None:
    #         index += 1
    #         return None
    #     node = TreeNode(nums[index])
    #     index += 1
    #     node.left = create_tree()
    #     node.right = create_tree()
    #     return node
    #
    # index = 0
    # return create_tree()


def list_linked_list(head: Optional[ListNode]) -> List[int]:
    result = []
    curr = head
    while curr:
        result.append(curr.val)
        curr = curr.next
    return result


def list_binary_tree(root: Optional[TreeNode]) -> List[Optional[int]]:
    if not root:
        return []
    result = []
    queue: List[Optional[TreeNode]] = [root]
    while queue:
        node = queue.pop(0)
        if node:
            result.append(node.val)
            queue.extend([node.left, node.right] if node else [])
        else:
            result.append(None)
    while result[-1] is None:
        result.pop()
    return result


def eq_linked_list(list1: Optional[ListNode], list2: Optional[ListNode]) -> bool:
    while list1 and list2:
        if list1.val != list2.val:
            return False
        list1, list2 = list1.next, list2.next
    return list1 is None and list2 is None


def eq_binary_tree(tree1: Optional[TreeNode], tree2: Optional[TreeNode]) -> bool:
    if tree1 is None and tree2 is None:
        return True
    if tree1 is None or tree2 is None:
        return False
    return (
        tree1.val == tree2.val and eq_binary_tree(tree1.left, tree2.left) and eq_binary_tree(tree1.right, tree2.right)
    )
