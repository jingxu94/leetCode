from typing import List


class Pro2001To2200:
    def __init__(self):
        pass

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
