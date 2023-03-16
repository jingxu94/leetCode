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
