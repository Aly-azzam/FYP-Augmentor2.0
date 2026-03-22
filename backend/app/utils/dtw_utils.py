"""Dynamic Time Warping utilities for multivariate time series."""

from math import inf

from app.utils.math_utils import euclidean_distance


def _validate_sequence_vectors(sequence: list[list[float]], name: str) -> int:
    """Validate that all vectors in a sequence have the same dimension."""
    if not sequence:
        return 0

    expected_dim = len(sequence[0])
    for index, vector in enumerate(sequence):
        if len(vector) != expected_dim:
            raise ValueError(
                f"{name} contains inconsistent vector sizes at index {index}: "
                f"expected {expected_dim}, got {len(vector)}"
            )
    return expected_dim


def compute_dtw(seq1: list[list[float]], seq2: list[list[float]]) -> dict:
    """Compute DTW distance and alignment path between two vector sequences.

    Returns:
        {
            "distance": float,
            "path": list[tuple[int, int]],
            "normalized_distance": float,
        }
    """
    n = len(seq1)
    m = len(seq2)

    if n == 0 and m == 0:
        return {
            "distance": 0.0,
            "path": [],
            "normalized_distance": 0.0,
        }

    if n == 0 or m == 0:
        return {
            "distance": inf,
            "path": [],
            "normalized_distance": inf,
        }

    dim1 = _validate_sequence_vectors(seq1, "seq1")
    dim2 = _validate_sequence_vectors(seq2, "seq2")
    if dim1 != dim2:
        raise ValueError(
            f"Vector dimension mismatch between seq1 and seq2: {dim1} != {dim2}"
        )

    cost = [[inf for _ in range(m + 1)] for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = euclidean_distance(seq1[i - 1], seq2[j - 1])
            cost[i][j] = dist + min(
                cost[i - 1][j],
                cost[i][j - 1],
                cost[i - 1][j - 1],
            )

    i, j = n, m
    path: list[tuple[int, int]] = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        neighbors = [
            (cost[i - 1][j], i - 1, j),
            (cost[i][j - 1], i, j - 1),
            (cost[i - 1][j - 1], i - 1, j - 1),
        ]
        _, i, j = min(neighbors, key=lambda item: item[0])

    while i > 0:
        path.append((i - 1, 0))
        i -= 1

    while j > 0:
        path.append((0, j - 1))
        j -= 1

    path.reverse()

    final_cost = cost[n][m]
    return {
        "distance": final_cost,
        "path": path,
        "normalized_distance": final_cost / (n + m),
    }
