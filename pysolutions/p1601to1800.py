import math
from typing import List


class Pro1601To1800:
    def __init__(self):
        pass

    def maximumWealth(self, accounts: List[List[int]]) -> int:
        # 1672.Richest Customer Wealth
        rich = []
        for customer in accounts:
            rich.append(sum(customer))
        return max(rich)

    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        # 1779.Find Nearest Point That Has the Same X or Y Coordinate
        index, smd = 0, math.inf
        for i, point in enumerate(points):
            if x == point[0] or y == point[1]:
                manhattan_distance = abs(x - point[0]) + abs(y - point[1])
                if manhattan_distance < smd:
                    smd = manhattan_distance
                    index = i
        if smd == math.inf:
            return -1
        return index

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        # 1790.Check if One String Swap Can Make Strings Equal
        ndiff = 0
        diff_ind = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                ndiff += 1
                diff_ind.append(i)
            if ndiff == 2:
                if s1[diff_ind[0]] != s2[diff_ind[1]] or s1[diff_ind[1]] != s2[diff_ind[0]]:
                    return False
            if ndiff > 2:
                return False
        if ndiff == 1:
            return False
        return True
