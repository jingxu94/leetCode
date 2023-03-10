import unittest

from pysolutions import Pro0201To0400

from .tools import set_ListNode


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TestP0201To0400(unittest.TestCase):
    @property
    def sl(self):
        return Pro0201To0400()
