"""Vector math in plain Python. No embedding or retrieval libraries.

Vectors must be non-empty sequences of finite real numbers. Paired vectors
must use the same dimensions AND coordinate meanings; only dimensions can
be checked here. Results outside the float range raise ValueError.
"""

from collections.abc import Sequence
from math import isfinite, sqrt

Vector = Sequence[float]


def _validate_vector(vector: Vector) -> None:
    if len(vector) == 0:
        raise ValueError("Vectors must not be empty.")
    if any(not isfinite(value) for value in vector):
        raise ValueError("Vector components must be finite.")


def _validate_pair(a: Vector, b: Vector) -> None:
    _validate_vector(a)
    _validate_vector(b)
    if len(a) != len(b):
        raise ValueError("Vectors must have the same number of dimensions.")


def dot_product(a: Vector, b: Vector) -> float:
    """Multiply matching coordinates and add: sum(a_i * b_i)."""
    _validate_pair(a, b)
    result = sum(x * y for x, y in zip(a, b, strict=True))
    if not isfinite(result):
        raise ValueError("Dot product exceeds the supported floating-point range.")
    return float(result)


def magnitude(vector: Vector) -> float:
    """Compute sqrt(sum(v_i**2)), rescaled to avoid intermediate overflow."""
    _validate_vector(vector)
    scale = max(abs(value) for value in vector)
    if scale == 0:
        return 0.0
    # Algebraically identical to sqrt(sum(v_i**2)); intermediate squares <= 1.
    result = scale * sqrt(sum((value / scale) ** 2 for value in vector))
    if not isfinite(result):
        raise ValueError("Magnitude exceeds the supported floating-point range.")
    return result


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Measure straight-line distance: sqrt(sum((a_i - b_i)**2))."""
    _validate_pair(a, b)
    return magnitude([x - y for x, y in zip(a, b, strict=True)])


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Measure direction alignment, from -1 to 1; undefined for zero vectors."""
    _validate_pair(a, b)
    length_a = magnitude(a)
    length_b = magnitude(b)
    if length_a == 0 or length_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    # Normalize before multiplying to avoid overflow in a dot b or |a| * |b|.
    unit_a = [value / length_a for value in a]
    unit_b = [value / length_b for value in b]
    score = dot_product(unit_a, unit_b)
    # Floating-point rounding can otherwise give 1.0000000000000002.
    return max(-1.0, min(1.0, score))
