"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable, List


class SegmentTree:
    """A Segment Tree data structure to perform efficient range queries and
    updates. This version supports operations like sum, min, max, and more,
    where each operation is associative.

    https://en.wikipedia.org/wiki/Segment_tree

    This is based on the OpenAI baselines implementation:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Can be used as a regular array that supports index arrays, but with two
    important differences:

        a) Setting an item's value is slightly slower.
            It is O(lg capacity) instead of O(1).
        b) User has access to an efficient ( O(log segment size) )
            `reduce` operation which reduces `operation` over
            a contiguous subsequence of items in the array.

    :param capacity: (int) Total size of the array - must be a power of two.
    :param operation: (Callable[[Any, Any], Any]) Operation for combining elements (e.g., sum, max) must form a
        mathematical group together with the set of possible values for array elements (i.e., be associative).
    :param init_value: (float) Initial value for the operation. E.g., float('-inf') for max and 0 for sum.
    """

    def __init__(self, capacity: int, operation: Callable,
                 init_value: float) -> None:
        assert (capacity > 0 and capacity & (capacity - 1)
                == 0), 'capacity must be positive and a power of 2.'
        self.capacity = capacity
        self.tree: List[float] = [init_value for _ in range(2 * capacity)]
        # self.tree 是一个长度为 2 * capacity 的列表，前一半存储内部节点，后一半存储叶子节点。
        self.operation = operation
        self.init_value = init_value

    def _operate_helper(self, start: int, end: int, node: int, node_start: int,
                        node_end: int) -> float:
        """Returns result of operation in segment. This method recursively
        divides the segment into smaller segments and applies the operation.

        通过递归二分区间，分别计算左右区间的结果，然后通过 operation（例如加法或最小值）将它们合并。

        :param start: (int) Start of the query range.
        :param end: (int) End of the query range.
        :param node: (int) Current node index in the tree.
        :param node_start: (int) Start of the segment corresponding to the current node.
        :param node_end: (int) End of the segment corresponding to the current node.
        :return: (float) Result of the operation over the specified range.
        """
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        elif start >= mid + 1:
            return self._operate_helper(start, end, 2 * node + 1, mid + 1,
                                        node_end)
        else:
            return self.operation(
                self._operate_helper(start, mid, 2 * node, node_start, mid),
                self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1,
                                     node_end),
            )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation` to a contiguous
        subsequence of the array.

        :param start: (int) Beginning of the subsequence.
        :param end: (int) End of the subsequence.
        :return: (float) Result of reducing self.operation over the specified range of array elements.

        功能：对数组某一段区间 [start, end] 应用 operation，例如求和或求最小值。

        参数：
            start 和 end：查询的区间范围。若 end 为 0，则默认查询整个数组。
        """
        if end <= 0:
            end += self.capacity
        end -= 1
        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree. 功能：更新树中某个索引的值，并相应地更新线段树的其他节点以保持树结构的正确性。

        参数：     idx：要更新的数组索引。     val：要设置的新值。 实现：
        从叶子节点（存储数组元素的位置）开始更新，然后向上回溯，更新每个父节点的值，确保整棵树的正确性。
        """
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx],
                                            self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity, 'Index out of range.'
        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """A specialized Segment Tree where the operation is summation.

    Inherits from the SegmentTree class and uses addition as the
    associative operation.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int) -> None:
        """Initialization.

        Args:
            capacity (int)
        """
        super().__init__(capacity=capacity,
                         operation=operator.add,
                         init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns the sum of the values in the range [start, end].

        :param start: (int) Start position of the reduction (must be >= 0).
        :param end: (int) End position of the reduction (must be < len(arr), can be None for len(arr) - 1).
        :return: (float) Reduction of SumSegmentTree.
        """
        return super().operate(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find the highest index `i` in the array such that.

            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        If array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        功能：根据累积和查找索引，返回使累积和不超过 prefixsum 的最大索引。
        参数：
            prefixsum：累积和的上限。
        实现：
            从根节点开始，通过比较左、右子节点的值进行二分查找，找到符合条件的叶子节点。
        """
        assert 0 <= prefixsum <= self.sum(
        ) + 1e-5, f'{prefixsum} out of range.'

        idx = 1
        while idx < self.capacity:  # While non-leaf
            left = 2 * idx
            right = 2 * idx + 1
            if self.tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """A specialized Segment Tree where the operation is finding the minimum
    value.

    Inherits from the SegmentTree class and uses the min function as the operation.


    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int) -> None:
        """Initialization.

        :param capacity: (int) Total size of the array - must be a power of two.
        """
        super().__init__(capacity=capacity,
                         operation=min,
                         init_value=float('inf'))

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ..., arr[end]).

        :param start: (int) Start position of the reduction (must be >= 0).
        :param end: (int) End position of the reduction (must be < len(arr), can be None for len(arr) - 1).
        :return: (float) Minimum value in the specified range.
        """
        return super().operate(start, end)
