from typing import Optional

from .utils import ListNode


class Pro0801To1000:
    def __init__(self):
        pass

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 876.Middle of the Linked List
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
