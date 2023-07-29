import unittest

from pysolutions import Pro0801To1000
from pysolutions.utils import create_binary_tree, create_linked_list, eq_linked_list


class TestP0801To1000(unittest.TestCase):
    @property
    def sl(self):
        return Pro0801To1000()

    def test_eventualSafeNodes(self):
        # 802.Find Eventual Safe States
        self.assertEqual(self.sl.eventualSafeNodes([[1, 2], [2, 3], [5], [0], [5], [], []]), [2, 4, 5, 6])
        self.assertEqual(self.sl.eventualSafeNodes([[1, 2, 3, 4], [1, 2], [3, 4], [0, 4], []]), [4])

    def test_soupServings(self):
        # 808.Soup Servings
        self.assertAlmostEqual(self.sl.soupServings(50), 0.625)
        self.assertAlmostEqual(self.sl.soupServings(100), 0.71875)
        self.assertAlmostEqual(self.sl.soupServings(101), 0.7421875)
        self.assertAlmostEqual(self.sl.soupServings(660295675), 0.9999999995343387)

    def test_numBusesToDestination(self):
        # 815.Bus Routes
        routes1, source1, target1 = [[1, 2, 7], [3, 6, 7]], 1, 6
        self.assertEqual(self.sl.numBusesToDestination(routes1, source1, target1), 2)
        self.assertEqual(self.sl.numBusesToDestination(routes1, 3, 3), 0)
        routes2 = [[7, 12], [4, 5, 15], [6], [15, 19], [9, 12, 13]]
        source2, target2 = 15, 12
        self.assertEqual(self.sl.numBusesToDestination(routes2, source2, target2), -1)

    def test_new21Game(self):
        # 837.New 21 Game
        self.assertAlmostEqual(self.sl.new21Game(10, 1, 10), 1.0)
        self.assertAlmostEqual(self.sl.new21Game(6, 1, 10), 0.6)
        self.assertAlmostEqual(self.sl.new21Game(21, 17, 10), 0.73277778)

    def test_numSimilarGroups(self):
        # 839.Similar String Groups
        strs1 = ["tars", "rats", "arts", "star"]
        strs2 = ["omv", "ovm"]
        self.assertEqual(self.sl.numSimilarGroups(strs1), 2)
        self.assertEqual(self.sl.numSimilarGroups(strs2), 1)

    def test_canVisitAllRooms(self):
        # 841.Keys and Rooms
        rooms1 = [[1], [2], [3], []]
        rooms2 = [[1, 3], [3, 0, 1], [2], [0]]
        self.assertTrue(self.sl.canVisitAllRooms(rooms1))
        self.assertFalse(self.sl.canVisitAllRooms(rooms2))

    def test_backspaceCompare(self):
        # 844.Backspace String Compare
        self.assertTrue(self.sl.backspaceCompare("ab#c", "ad#c"))
        self.assertTrue(self.sl.backspaceCompare("ab##", "c#d#"))
        self.assertFalse(self.sl.backspaceCompare("a#c", "b"))

    def test_peakIndexInMountainArray(self):
        # 852.Peak index in a Mountain Array
        self.assertEqual(self.sl.peakIndexInMountainArray([0, 1, 0]), 1)
        self.assertEqual(self.sl.peakIndexInMountainArray([0, 2, 1, 0]), 1)
        self.assertEqual(self.sl.peakIndexInMountainArray([0, 10, 5, 2]), 1)

    def test_buddyStrings(self):
        # 859.Buddy Strings
        self.assertTrue(self.sl.buddyStrings("ab", "ba"))
        self.assertFalse(self.sl.buddyStrings("ab", "ab"))
        self.assertTrue(self.sl.buddyStrings("aa", "aa"))
        self.assertTrue(self.sl.buddyStrings("aaaaaaabc", "aaaaaaacb"))
        self.assertFalse(self.sl.buddyStrings("", "aa"))

    def test_lemonadeChange(self):
        # 860.Lemonade Change
        self.assertTrue(self.sl.lemonadeChange([5, 5, 5, 10, 20]))
        self.assertTrue(self.sl.lemonadeChange([5, 5, 10]))
        self.assertFalse(self.sl.lemonadeChange([10, 10]))
        self.assertFalse(self.sl.lemonadeChange([5, 20, 20, 20, 20]))

    def test_distanceK(self):
        # 863.All Nodes Distance K in Binary Tree
        root1 = create_binary_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
        target1 = root1.left
        self.assertEqual(self.sl.distanceK(root1, target1, 2), [1, 7, 4])
        root2 = create_binary_tree([0, 1, None, 3, 2])
        target2 = root2.left.right
        self.assertEqual(self.sl.distanceK(root2, target2, 1), [1])

    def test_shortestPathAllKeys(self):
        # 864.Shortest Path to Get All Keys
        grid1 = ["@.a.#", "###.#", "b.A.B"]
        grid2 = ["@..aA", "..B#.", "....b"]
        grid3 = ["@...a", ".###A", "b.BCc"]
        grid4 = ["@Aa"]
        self.assertEqual(self.sl.shortestPathAllKeys(grid1), 8)
        self.assertEqual(self.sl.shortestPathAllKeys(grid2), 6)
        self.assertEqual(self.sl.shortestPathAllKeys(grid3), 10)
        self.assertEqual(self.sl.shortestPathAllKeys(grid4), -1)

    def test_leafSimilar(self):
        # 872.Leaf-Similar Trees
        root1 = create_binary_tree([3, 5, 1, 6, 2, 9, 8, None, None, 7, 4])
        root2 = create_binary_tree([3, 5, 1, 6, 7, 4, 2, None, None, None, None, None, None, 9, 8])
        self.assertTrue(self.sl.leafSimilar(root1, root2))
        root1 = create_binary_tree([1, 2, 3])
        root2 = create_binary_tree([1, 3, 2])
        self.assertFalse(self.sl.leafSimilar(root1, root2))
        self.assertTrue(self.sl.leafSimilar(None, None))

    def test_minEatingSpeed(self):
        # 875.Koko Eating Bananas
        piles1 = [3, 6, 7, 11]
        piles2 = [30, 11, 23, 4, 20]
        piles3 = [30, 11, 23, 4, 20]
        h1, h2, h3 = 8, 5, 6
        expres1, expres2, expres3 = 4, 30, 23
        self.assertEqual(self.sl.minEatingSpeed(piles1, h1), expres1)
        self.assertEqual(self.sl.minEatingSpeed(piles2, h2), expres2)
        self.assertEqual(self.sl.minEatingSpeed(piles3, h3), expres3)

    def test_middleNode(self):
        # 876.Middle of the Linked List
        head = create_linked_list([1, 2, 3, 4, 5])
        expected = create_linked_list([3, 4, 5])
        ans = self.sl.middleNode(head)
        self.assertTrue(eq_linked_list(ans, expected))
        head = create_linked_list([1, 2, 3, 4, 5, 6])
        expected = create_linked_list([4, 5, 6])
        ans = self.sl.middleNode(head)
        self.assertTrue(eq_linked_list(ans, expected))
        self.assertIsNone(self.sl.middleNode(None))

    def test_profitableSchemes(self):
        # 879.Profitable Schemes
        n1, minProfit1, group1, profit1 = 5, 3, [2, 2], [2, 3]
        n2, minProfit2, group2, profit2 = 10, 5, [2, 3, 5], [6, 7, 8]
        self.assertEqual(self.sl.profitableSchemes(n1, minProfit1, group1, profit1), 2)
        self.assertEqual(self.sl.profitableSchemes(n2, minProfit2, group2, profit2), 7)

    def test_numRescueBoats(self):
        # 881.Boats to Save People
        self.assertEqual(self.sl.numRescueBoats([1, 2], 3), 1)
        self.assertEqual(self.sl.numRescueBoats([3, 2, 2, 1], 3), 3)
        self.assertEqual(self.sl.numRescueBoats([3, 5, 3, 4], 5), 4)

    def test_allPossibleFBT(self):
        # 894.All Possible Full Binary Trees
        n1 = 7
        n2 = 3
        ans1 = self.sl.allPossibleFBT(n1)
        ans2 = self.sl.allPossibleFBT(n2)
        self.assertEqual(len(ans1), 5)
        self.assertEqual(len(ans2), 1)
        self.assertEqual(self.sl.allPossibleFBT(2), [])

    def test_isMonotonic(self):
        # 896.Monotonic Array
        self.assertTrue(self.sl.isMonotonic([1, 2, 2, 3]))
        self.assertTrue(self.sl.isMonotonic([6, 5, 4, 4]))
        self.assertFalse(self.sl.isMonotonic([1, 3, 2]))

    def test_smallestRangeII(self):
        # 910.Smallest Range II
        self.assertEqual(self.sl.smallestRangeII([1], 0), 0)
        self.assertEqual(self.sl.smallestRangeII([0, 10], 2), 6)
        self.assertEqual(self.sl.smallestRangeII([1, 3, 6], 3), 3)

    def test_sortArray(self):
        # 912.Sort an Array
        nums1 = [5, 2, 3, 1]
        nums2 = [5, 1, 1, 2, 0, 0]
        expres1 = [1, 2, 3, 5]
        expres2 = [0, 0, 1, 1, 2, 5]
        self.assertEqual(self.sl.sortArray(nums1), expres1)
        self.assertEqual(self.sl.sortArray(nums2), expres2)

    def test_maxSubarraySumCircular(self):
        # 918.Maximum Sum Circular Subarray
        self.assertEqual(self.sl.maxSubarraySumCircular([1, -2, 3, -2]), 3)
        self.assertEqual(self.sl.maxSubarraySumCircular([5, -3, 5]), 10)
        self.assertEqual(self.sl.maxSubarraySumCircular([-3, -2, -3]), -2)

    def test_minFallingPathSum(self):
        # 931.Minimum Falling Path Sum
        A1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        A2 = [[-19, 57], [-40, -5]]
        self.assertEqual(self.sl.minFallingPathSum(A1), 12)
        self.assertEqual(self.sl.minFallingPathSum(A2), -59)

    def test_shortestBridge(self):
        # 934.Shortest Bridge
        grid = [[0, 1], [1, 0]]
        self.assertEqual(self.sl.shortestBridge(grid), 1)
        grid = [[0, 1, 0], [0, 0, 0], [0, 0, 1]]
        self.assertEqual(self.sl.shortestBridge(grid), 2)
        grid = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
        self.assertEqual(self.sl.shortestBridge(grid), 1)

    def test_validateStackSequences(self):
        # 946.Validate Stack Sequences
        pushed1 = [1, 2, 3, 4, 5]
        popped1 = [4, 5, 3, 2, 1]
        pushed2 = [1, 2, 3, 4, 5]
        popped2 = [4, 3, 5, 1, 2]
        self.assertTrue(self.sl.validateStackSequences(pushed1, popped1))
        self.assertFalse(self.sl.validateStackSequences(pushed2, popped2))

    def test_isAlienSorted(self):
        # 953.Verifying an Alien Dictionary
        words = ["hello", "leetcode"]
        order = "hlabcdefgijkmnopqrstuvwxyz"
        self.assertTrue(self.sl.isAlienSorted(words, order))
        self.assertTrue(self.sl.isAlienSorted(["hello"], order))
        words = ["word", "world", "row"]
        order = "worldabcefghijkmnpqstuvxyz"
        self.assertFalse(self.sl.isAlienSorted(words, order))
        words = ["apple", "app"]
        order = "abcdefghijklmnopqrstuvwxyz"
        self.assertFalse(self.sl.isAlienSorted(words, order))
        self.assertTrue(self.sl.isAlienSorted(["app", "apple"], order))

    def test_tallestBillboard(self):
        # 956.Tallest Billboard
        self.assertEqual(self.sl.tallestBillboard([1, 2, 3, 6]), 6)
        self.assertEqual(self.sl.tallestBillboard([1, 2, 3, 4, 5, 6]), 10)
        self.assertEqual(self.sl.tallestBillboard([1, 2]), 0)

    def test_isCompleteTree(self):
        # 958.Check Completeness of a Binary Tree
        root1 = create_binary_tree([1, 2, 3, 4, 5, 6])
        root2 = create_binary_tree([1, 2, 3, 4, 5, None, 7])
        self.assertTrue(self.sl.isCompleteTree(root1))
        self.assertFalse(self.sl.isCompleteTree(root2))
        self.assertTrue(self.sl.isCompleteTree(None))

    def test_kClosest(self):
        # 973.K Closest Points to Origin
        self.assertEqual(self.sl.kClosest([[1, 3], [-2, 2]], 1), [[-2, 2]])
        self.assertEqual(self.sl.kClosest([[3, 3], [5, -1], [-2, 4]], 2), [[3, 3], [-2, 4]])

    def test_largestPerimeter(self):
        # 976.Largest Perimeter Triangle
        self.assertEqual(self.sl.largestPerimeter([2, 1, 2]), 5)
        self.assertEqual(self.sl.largestPerimeter([1, 2, 1, 10]), 0)

    def test_sortedSquares(self):
        # 977.Squares of a Sorted Array
        self.assertListEqual(self.sl.sortedSquares([-4, -1, 0, 3, 10]), [0, 1, 9, 16, 100])
        self.assertListEqual(self.sl.sortedSquares([-7, -3, 2, 3, 11]), [4, 9, 9, 49, 121])

    def test_mincostTickets(self):
        # 983.Minimum Cost For Tickets
        self.assertEqual(self.sl.mincostTickets([1, 4, 6, 7, 8, 20], [2, 7, 15]), 11)
        self.assertEqual(self.sl.mincostTickets([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 31], [2, 7, 15]), 17)

    def test_intervalIntersection(self):
        # 986.Interval List Intersections
        firstList = [[0, 2], [5, 10], [13, 23], [24, 25]]
        secondList = [[1, 5], [8, 12], [15, 24], [25, 26]]
        ans = [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]
        self.assertEqual(self.sl.intervalIntersection(firstList, secondList), ans)

    def test_addToArrayForm(self):
        # 989.Add to Array-Form of Integer
        self.assertEqual(self.sl.addToArrayForm([1, 2, 0, 0], 34), [1, 2, 3, 4])
        self.assertEqual(self.sl.addToArrayForm([2, 7, 4], 181), [4, 5, 5])
        self.assertEqual(self.sl.addToArrayForm([2, 1, 5], 806), [1, 0, 2, 1])

    def test_orangesRotting(self):
        # 994.Rotting Oranges
        self.assertEqual(self.sl.orangesRotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]), 4)
        self.assertEqual(self.sl.orangesRotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]), -1)
        self.assertEqual(self.sl.orangesRotting([[0, 2]]), 0)

    def test_findJudge(self):
        # 997.Find the Town Judge
        self.assertEqual(self.sl.findJudge(2, [[1, 2]]), 2)
        self.assertEqual(self.sl.findJudge(1, [[1, 2]]), 1)
        self.assertEqual(self.sl.findJudge(3, [[1, 3], [2, 3]]), 3)
        self.assertEqual(self.sl.findJudge(3, [[1, 3], [2, 3], [3, 1]]), -1)
