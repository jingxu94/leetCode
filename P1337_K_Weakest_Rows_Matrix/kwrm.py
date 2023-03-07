class Solution:
    def kWeakestRows(self, mat: list[list[int]], k: int) -> list[int]:
        soldiers = []
        for row in range(len(mat)):
            soldiers.append(sum(mat[row]))
        arg_soldiers = sorted(range(len(soldiers)), key=soldiers.__getitem__)
        return arg_soldiers[:k]


if __name__ == "__main__":
    mat = [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    k = 3
    print(Solution().kWeakestRows(mat, k))
