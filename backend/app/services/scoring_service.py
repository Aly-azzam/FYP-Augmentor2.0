"""Scoring Service — convert DTW distance into a score /100."""


class ScoringService:
    """Compute a simple score from normalized DTW distance."""

    def compute_score(self, normalized_distance: float) -> dict[str, float]:
        """Convert normalized DTW distance into a 0..100 score."""
        safe_distance = max(0.0, float(normalized_distance))
        score = max(0.0, 100.0 * (1.0 - safe_distance))
        return {
            "score": score,
            "normalized_distance": safe_distance,
        }
