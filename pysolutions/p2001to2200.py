from typing import List


class Pro2001To2200:
    def __init__(self):
        pass

    def longestPalindrome(self, words: List[str]) -> int:
        # 2131.Longest Palindrome by Concatenating Two Letter Words
        alphabet_size = 26
        count = [[0 for j in range(alphabet_size)] for i in range(alphabet_size)]
        for word in words:
            count[ord(word[0]) - ord("a")][ord(word[1]) - ord("a")] += 1
        answer = 0
        central = False
        for i in range(alphabet_size):
            if count[i][i] % 2 == 0:
                answer += count[i][i]
            else:
                answer += count[i][i] - 1
                central = True
            for j in range(i + 1, alphabet_size):
                answer += 2 * min(count[i][j], count[j][i])
        if central:
            answer += 1
        return 2 * answer

    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        # 2187.Minimum Time to Complete
        def cal_trips(time: List[int], now: int):
            trips = 0
            for t in time:
                trips += now // t
            return trips

        left = 0
        right = time[0] * totalTrips
        while left < right:
            mid = (left + right) // 2
            trips = cal_trips(time, mid)
            if trips < totalTrips:
                left = mid + 1
            else:
                right = mid
        return left
