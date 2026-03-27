"""Deterministic temporal alignment using Dynamic Time Warping (DTW)."""

from __future__ import annotations

import math
from typing import Any


def _extract_sequence_vectors(motion_data: dict[str, Any]) -> list[list[float]]:
    frames = motion_data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Motion input must contain a non-empty 'frames' list.")

    sequence: list[list[float]] = []
    expected_dim: int | None = None

    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be a dictionary.")

        vector = frame.get("flattened_feature_vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError("Each frame must contain a non-empty 'flattened_feature_vector'.")

        cast_vector = [float(value) for value in vector]
        if expected_dim is None:
            expected_dim = len(cast_vector)
        elif len(cast_vector) != expected_dim:
            raise ValueError("All flattened_feature_vector entries must have the same dimension.")

        sequence.append(cast_vector)

    return sequence


def _euclidean_distance(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Cannot compute distance for vectors with different dimensions.")
    squared_sum = sum((a - b) ** 2 for a, b in zip(vector_a, vector_b))
    return math.sqrt(squared_sum)


def align_sequences(
    expert_motion: dict[str, Any],
    learner_motion: dict[str, Any],
) -> dict[str, Any]:
    """Align two motion sequences with DTW and return path/cost metadata."""
    expert_sequence = _extract_sequence_vectors(expert_motion)
    learner_sequence = _extract_sequence_vectors(learner_motion)

    n = len(expert_sequence)
    m = len(learner_sequence)

    cost: list[list[float]] = [[float("inf")] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            local_distance = _euclidean_distance(expert_sequence[i], learner_sequence[j])
            if i == 0 and j == 0:
                cost[i][j] = local_distance
            elif i == 0:
                cost[i][j] = local_distance + cost[i][j - 1]
            elif j == 0:
                cost[i][j] = local_distance + cost[i - 1][j]
            else:
                cost[i][j] = local_distance + min(
                    cost[i - 1][j],
                    cost[i][j - 1],
                    cost[i - 1][j - 1],
                )

    i = n - 1
    j = m - 1
    reversed_path: list[list[int]] = [[i, j]]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = (
                (cost[i - 1][j], i - 1, j),
                (cost[i][j - 1], i, j - 1),
                (cost[i - 1][j - 1], i - 1, j - 1),
            )
            _, next_i, next_j = min(candidates, key=lambda item: item[0])
            i, j = next_i, next_j
        reversed_path.append([i, j])

    alignment_path = list(reversed(reversed_path))
    aligned_pairs = [
        {"expert_index": expert_index, "learner_index": learner_index}
        for expert_index, learner_index in alignment_path
    ]
    total_cost = float(cost[n - 1][m - 1])
    path_length = len(alignment_path)
    normalized_cost = total_cost / path_length if path_length > 0 else 0.0

    return {
        "alignment_path": alignment_path,
        "aligned_pairs": aligned_pairs,
        "dtw_total_cost": total_cost,
        "dtw_normalized_cost": normalized_cost,
        "path_length": path_length,
    }
