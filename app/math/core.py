import numpy as np


def calculate_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """Calculates the mean vector (centroid) from a list of embeddings."""
    if not vectors:
        raise ValueError("Cannot calculate centroid from an empty list of vectors.")
    return np.mean(vectors, axis=0)


def debias_axis(
    v_low: np.ndarray, v_high: np.ndarray, v_neutral: np.ndarray
) -> np.ndarray:
    """
    Calculates 1D score axis and applies orthogonalization to remove neutral/corporate projections.
    """
    axis_raw = v_high - v_low
    n_norm_sq = np.dot(v_neutral, v_neutral)

    # prevent division by 0
    if n_norm_sq == 0:
        return axis_raw

    projection_scalar = np.dot(axis_raw, v_neutral) / n_norm_sq
    return axis_raw - (projection_scalar * v_neutral)


def project_scalar(target: np.ndarray, v_low: np.ndarray, axis: np.ndarray) -> float:
    """
    Projects a target vector onto an axis.
    """
    d_norm_sq = np.dot(axis, axis)
    if d_norm_sq == 0:
        return 0.0

    diff = target - v_low
    projection = np.dot(diff, axis) / d_norm_sq

    return float(projection)


def softmin_weighted_average(scores: list[float], temperature: float = 0.2) -> float:
    """
    Calculates the expected value of scores weighed by Softmin function; lower scores are weighed exponentially more than higher ones.
    """
    if not scores:
        return np.nan

    scores_array = np.array(scores)

    shifted_scores = scores_array - np.min(scores_array)
    weights = np.exp(-shifted_scores / temperature)

    return float(np.dot(weights, scores_array) / weights.sum())
