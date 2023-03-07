# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
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
