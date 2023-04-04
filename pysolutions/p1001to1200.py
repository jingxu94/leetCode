from collections import Counter
from typing import List


class Pro1001To1200:
    def __init__(self):
        pass

    def commonChars(self, words: List[str]) -> List[str]:
        # 1002.Find Common Characters
        flag = Counter(words[0])
        if len(words) == 1:
            return [i for i in flag.elements()]
        for word in words[1:]:
            check = Counter(word)
            for key in flag.keys():
                if check.get(key, 0) < flag[key]:
                    flag[key] = check[key]
        return [i for i in flag.elements()]

    def lastStoneWeight(self, stones: List[int]) -> int:
        # 1046.Last Stone Weight
        while len(stones) > 1:
            stones.sort()
            y = stones.pop()
            x = stones.pop()
            if y != x:
                stones.append(y - x)
        return stones[0] if stones else 0

    def tribonacci(self, n: int) -> int:
        # 1137.N-th Tribonacci Number
        if n == 0 or n == 1:
            return n
        elif n == 2:
            return 1
        else:
            fst, sec, trd = 0, 1, 1
            for _ in range(n - 2):
                fst, sec, trd = sec, trd, fst + sec + trd
            return trd
