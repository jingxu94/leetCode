from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def set_ListNode(numbers: List[int]):
    result = ListNode()
    curr = result
    for num in numbers:
        curr.next = ListNode(num)
        curr = curr.next
    return result.next
