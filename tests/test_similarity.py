"""Check mathematical identities, known values, and invalid vector inputs."""

import math

import pytest

from rag_lab.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    magnitude,
)


def test_dot_product_known_value() -> None:
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_dot_product_keeps_sign() -> None:
    assert dot_product([1, -2], [3, 4]) == -5


def test_magnitude_known_value() -> None:
    assert magnitude([3, 4]) == pytest.approx(5)


def test_zero_vector_has_zero_magnitude() -> None:
    assert magnitude([0, 0]) == 0


def test_magnitude_avoids_intermediate_overflow() -> None:
    assert magnitude([3e200, 4e200]) == pytest.approx(5e200)


def test_magnitude_avoids_intermediate_underflow() -> None:
    assert magnitude([3e-200, 4e-200]) == pytest.approx(5e-200, abs=0)


def test_euclidean_distance_known_value() -> None:
    assert euclidean_distance([1, 2], [4, 6]) == pytest.approx(5)


def test_identical_vectors_have_zero_distance() -> None:
    assert euclidean_distance([3, 4], [3, 4]) == 0


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 1),
        ([1, 2], [2, 4], 1),
        ([1, 0], [0, 1], 0),
        ([1, 0], [-1, 0], -1),
        ([1, 0], [4, 1], 4 / math.sqrt(17)),
        ([3e200, 4e200], [3e200, 4e200], 1),
        ([3e-200, 4e-200], [3e-200, 4e-200], 1),
    ],
)
def test_cosine_known_directions(a, b, expected) -> None:
    assert cosine_similarity(a, b) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [dot_product, euclidean_distance, cosine_similarity])
def test_pairwise_metrics_are_symmetric(metric) -> None:
    assert metric([1, 3, -2], [2, -4, 5]) == pytest.approx(
        metric([2, -4, 5], [1, 3, -2])
    )


def test_positive_scaling_changes_dot_and_distance_but_not_cosine() -> None:
    query, document, scaled = [1, 0], [4, 1], [8, 2]
    assert cosine_similarity(query, document) == pytest.approx(
        cosine_similarity(query, scaled)
    )
    assert dot_product(query, scaled) == 2 * dot_product(query, document)
    assert euclidean_distance(query, document) != euclidean_distance(query, scaled)


@pytest.mark.parametrize("metric", [dot_product, euclidean_distance, cosine_similarity])
@pytest.mark.parametrize(("a", "b"), [([], []), ([1], []), ([1, 2], [1])])
def test_invalid_shapes_are_rejected(metric, a, b) -> None:
    with pytest.raises(ValueError):
        metric(a, b)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_components_are_rejected(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        magnitude([invalid, 1])
    for metric in (dot_product, euclidean_distance, cosine_similarity):
        with pytest.raises(ValueError, match="finite"):
            metric([invalid, 1], [1, 2])
        with pytest.raises(ValueError, match="finite"):
            metric([1, 2], [invalid, 1])


@pytest.mark.parametrize(
    ("a", "b"), [([0, 0], [1, 0]), ([1, 0], [0, 0]), ([0, 0], [0, 0])]
)
def test_cosine_rejects_zero_vectors(a, b) -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity(a, b)


def test_magnitude_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="empty"):
        magnitude([])


def test_unrepresentable_dot_product_is_rejected() -> None:
    with pytest.raises(ValueError, match="range"):
        dot_product([1e308], [1e308])


def test_unrepresentable_magnitude_is_rejected() -> None:
    with pytest.raises(ValueError, match="range"):
        magnitude([1.7e308, 1.7e308])
