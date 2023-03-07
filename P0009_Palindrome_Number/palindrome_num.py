class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        x_raw = x
        x_flip = 0
        while x // 10 > 0:
            x_flip = x_flip * 10 + (x % 10)
            x = x // 10
        x_flip = x_flip * 10 + (x % 10)
        return x_raw == x_flip
