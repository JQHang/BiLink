"""
Array utility functions.
"""

import numpy as np


def grouped_arange(counts: np.ndarray, start: int = 1) -> np.ndarray:
    """
    Generate grouped arange sequence.

    For each group with count n, generates [start, start+1, ..., start+n-1].

    Args:
        counts: Array of group sizes
        start: Starting value for each group (default: 1)

    Returns:
        Concatenated arange sequences for all groups

    Examples:
        >>> grouped_arange(np.array([3, 2, 4]), start=1)
        array([1, 2, 3, 1, 2, 1, 2, 3, 4])

        >>> grouped_arange(np.array([3, 2, 4]), start=0)
        array([0, 1, 2, 0, 1, 0, 1, 2, 3])
    """
    # Total number of elements across all groups
    total = counts.sum()

    # Initialize with all ones: [1, 1, 1, 1, 1, ...]
    ones = np.ones(total, dtype=np.int64)

    # At each group boundary, subtract previous group's count to reset cumsum
    # Example: counts=[3,2,4], boundaries at [3,5]
    # ones[3] -= 3, ones[5] -= 2
    # ones becomes [1,1,1,-2,1,-1,1,1,1]
    ones[np.cumsum(counts[:-1])] -= counts[:-1]

    # Cumsum gives [1,2,3,1,2,1,2,3,4], adjust by (start-1) for different start
    return np.cumsum(ones) + (start - 1)
