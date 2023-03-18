class Pro2401To2600:
    def __init__(self) -> None:
        pass

    def splitNum(self, num: int) -> int:
        # 2578.Split With Minimum Sum
        digits = list(int(n) for n in str(num))
        digits.sort()
        digits.reverse()
        power1, power2 = len(digits) // 2, len(digits) - len(digits) // 2
        ans = 0
        for _ in range(power1):
            ans += 10 ** (power2 - 1) * digits[-1]
            digits.pop()
            ans += 10 ** (power1 - 1) * digits[-1]
            digits.pop()
            power1, power2 = power1 - 1, power2 - 1
        if power2:
            ans += digits[0]
        return ans
