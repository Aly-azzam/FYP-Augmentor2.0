"""Small reusable helpers for the Evaluation Engine."""

from collections.abc import Iterable


def round_metric(value: float, digits: int = 4) -> float:
    """Round a metric value to a consistent number of decimal places."""
    return round(value, digits)


def safe_average(values: Iterable[float]) -> float:
    """Return the average of numeric values, or 0.0 when empty."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value between a minimum and maximum bound."""
    return max(minimum, min(value, maximum))
