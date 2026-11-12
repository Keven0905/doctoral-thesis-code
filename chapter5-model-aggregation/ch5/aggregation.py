import numpy as np


def update_difficulty(probabilities, labels, difficulty, coefficient=1.0, epsilon=1e-12):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if len(labels) == 0 or difficulty.shape != (len(labels),):
        raise ValueError("Difficulty must have one entry per training sample.")
    if not np.isfinite(difficulty).all() or (difficulty < 0).any() or difficulty.sum() <= 0:
        raise ValueError("Invalid sample difficulty distribution.")
    difficulty = difficulty / difficulty.sum()
    errors = 1.0 - probabilities[np.arange(len(labels)), labels]
    mean = float(errors.mean())
    std = float(errors.std(ddof=0))
    hard = errors > mean + coefficient * std
    raw_rate = float(difficulty[hard].sum())
    rate = float(np.clip(raw_rate, epsilon, 1.0 - epsilon))
    rho = float(-np.log(rate))
    next_difficulty = difficulty * np.where(hard, 1.0, rate)
    next_difficulty /= next_difficulty.sum()
    stats = {"mean_error": mean, "std_error": std,
             "hard_threshold": mean + coefficient * std,
             "hard_count": int(hard.sum()), "raw_error_rate": raw_rate,
             "clipped_error_rate": rate, "rho": rho}
    return next_difficulty, stats, hard


def fuzzy_weights(mean_errors, deterministic_weights, epsilon=1e-12):
    errors = np.clip(np.asarray(mean_errors, dtype=np.float64), 0, 1)
    location = errors * 5.0
    left = np.minimum(np.floor(location).astype(int), 4)
    fraction = location - left
    high = 1.0 - left / 5.0
    low = 1.0 - (left + 1) / 5.0
    raw = (1.0 - fraction) * high + fraction * low
    fallback = bool(raw.sum() <= epsilon)
    if fallback:
        raw = np.asarray(deterministic_weights, dtype=np.float64)
    return normalize_weights(raw, epsilon), {"levels": "ABCDEF", "input_domain": [0, 1],
        "output_domain": [0, 1], "mean_errors": errors.tolist(),
        "left_level": left.tolist(), "right_membership": fraction.tolist(),
        "fallback_to_M7": fallback, "raw_weights": raw.tolist()}


def normalize_weights(weights, epsilon=1e-12):
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= epsilon:
        raise ValueError("Aggregation weights must be finite, nonnegative and have a positive sum.")
    return weights / weights.sum()


def aggregate(predictions, weights):
    weights = normalize_weights(weights)
    if len(predictions) != len(weights):
        raise ValueError("Prediction and weight counts differ.")
    probabilities = np.zeros_like(predictions[0], dtype=np.float64)
    for weight, prediction in zip(weights, predictions):
        probabilities += weight * prediction
    return probabilities / probabilities.sum(axis=1, keepdims=True)
