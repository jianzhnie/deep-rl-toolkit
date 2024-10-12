from typing import Optional, Union

import numpy as np
from numba import njit


class SegmentTree:
    """Implementation of Segment Tree.

    The segment tree stores an array ``arr`` with size ``n``. It supports value
    update and fast query of the sum for the interval ``[left, right)`` in
    O(log n) time. The detailed procedure is as follows:

    1. Pad the array to have length of power of 2, so that leaf nodes in the \
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.

    :param int size: the size of segment tree.
    """

    def __init__(self, size: int) -> None:
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._value = np.zeros([bound * 2])
        self._compile()

    def __len__(self) -> int:
        return self._size

    def __getitem__(self,
                    index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Return self[index]."""
        return self._value[index + self._bound]

    def __setitem__(self, index: Union[int, np.ndarray],
                    value: Union[float, np.ndarray]) -> None:
        """Update values in segment tree.

        Duplicate values in ``index`` are handled by numpy: later index
        overwrites previous ones.
        ::

            >>> a = np.array([1, 2, 3, 4])
            >>> a[[0, 1, 0, 1]] = [4, 5, 6, 7]
            >>> print(a)
            [6 7 3 4]
        """
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(0 <= index) and np.all(index < self._size)
        _setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return _reduce(self._value, start + self._bound - 1, end + self._bound)

    def get_prefix_sum_idx(
            self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        r"""Find the index with given value.

        Return the minimum index for each ``v`` in ``value`` so that
        :math:`v \le \mathrm{sums}_i`, where
        :math:`\mathrm{sums}_i = \sum_{j = 0}^{i} \mathrm{arr}_j`.

        .. warning::

            Please make sure all of the values inside the segment tree are
            non-negative when using this function.
        """
        assert np.all(value >= 0.0) and np.all(value < self._value[1])
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        _setitem(f64, i64, f64)
        _setitem(f64, i64, f32)
        _reduce(f64, 0, 1)
        _get_prefix_sum_idx(f64, 1, f64)
        _get_prefix_sum_idx(f32, 1, f64)


class MinSegmentTree:
    """Implementation of Segment Tree.

    The segment tree stores an array ``arr`` with size ``n``. It supports value
    update and fast query of the sum for the interval ``[left, right)`` in
    O(log n) time. The detailed procedure is as follows:

    1. Pad the array to have length of power of 2, so that leaf nodes in the \
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.

    :param int size: the size of segment tree.
    """

    def __init__(self, size: int) -> None:
        """Initialize the SegmentTree.

        :param size: (int) The size of the segment tree.
        """
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._value = np.full([bound * 2], np.inf)
        # Initialize with infinity for min operations
        self._compile()

    def __len__(self) -> int:
        """Return the size of the segment tree.

        :return: (int) The size of the segment tree.
        """
        return self._size

    def __getitem__(self,
                    index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Return self[index].

        :param index: (int or np.ndarray) The index or indices to retrieve.
        :return: (float or np.ndarray) The value(s) at the specified index(es).
        """
        return self._value[index + self._bound]

    def __setitem__(self, index: Union[int, np.ndarray],
                    value: Union[float, np.ndarray]) -> None:
        """Update values in segment tree.

        Duplicate values in ``index`` are handled by numpy: later index
        overwrites previous ones.
        ::

            >>> a = np.array([1, 2, 3, 4])
            >>> a[[0, 1, 0, 1]] = [4, 5, 6, 7]
            >>> print(a)
            [6 7 3 4]

        :param index: (int or np.ndarray) The index or indices to update.
        :param value: (float or np.ndarray) The new value(s) to set.
        """
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(0 <= index) and np.all(
            index < self._size), 'Index out of range.'
        _min_setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation(value[start:end]).

        :param start: (int) The start index of the interval (default is 0).
        :param end: (Optional[int]) The end index of the interval (default is None, which means the end of the array).
        :return: (float) The sum of the values in the specified interval.
        """
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return _min_reduce(self._value, start + self._bound - 1,
                           end + self._bound)

    def _compile(self) -> None:
        """Compile the Numba functions."""
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        _min_setitem(f64, i64, f64)
        _min_setitem(f64, i64, f32)
        _min_reduce(f64, 0, 1)


@njit
def _setitem(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = tree[index * 2] + tree[index * 2 + 1]


@njit
def _reduce(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""
    # nodes in (start, end) should be aggregated
    result = 0.0
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result += tree[start + 1]
        start //= 2
        if end % 2 == 1:
            result += tree[end - 1]
        end //= 2
    return result


@njit
def _min_setitem(tree: np.ndarray, index: np.ndarray,
                 value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024.

    :param tree: (np.ndarray) The segment tree array.
    :param index: (np.ndarray) The index or indices to update.
    :param value: (np.ndarray) The new value(s) to set.
    """
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = np.minimum(tree[index * 2], tree[index * 2 + 1])


@njit
def _min_reduce(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005.

    :param tree: (np.ndarray) The segment tree array.
    :param start: (int) The start index of the interval.
    :param end: (int) The end index of the interval.
    :return: (float) The minimum value in the specified interval.
    """
    result = np.inf
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result = np.minimum(result, tree[start + 1])
        start //= 2
        if end % 2 == 1:
            result = np.minimum(result, tree[end - 1])
        end //= 2
    return result


@njit
def _get_prefix_sum_idx(value: np.ndarray, bound: int,
                        sums: np.ndarray) -> np.ndarray:
    """Numba version (v0.51), 5x speed up with size=100000 and bsz=64.

    vectorized np: 0.0923 (numpy best) -> 0.024 (now)
    for-loop: 0.2914 -> 0.019 (but not so stable)
    """
    index = np.ones(value.shape, dtype=np.int64)
    while index[0] < bound:
        index *= 2
        lsons = sums[index]
        direct = lsons < value
        value -= lsons * direct
        index += direct
    index -= bound
    return index


if __name__ == '__main__':
    # 创建一个大小为 8 的最小分割树
    min_tree = MinSegmentTree(8)

    # 更新某些值
    min_tree[0] = 5
    min_tree[1] = 3
    min_tree[2] = 7
    min_tree[3] = 9
    min_tree[4] = 6

    # 查询 [0, 4) 范围内的最小值
    print(min_tree.reduce(0, 4))  # 输出: 3

    # 更新某个值后查询
    min_tree[1] = 10
    print(min_tree.reduce(0, 4))  # 输出: 5
