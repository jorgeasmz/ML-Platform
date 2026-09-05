import numpy as np
import pytest

from registry.drift import (
    Signal,
    categorical_index,
    compare,
    insufficient,
    numerical_index,
)

RNG = np.random.default_rng(0)


def test_an_identical_numerical_sample_does_not_move():
    values = RNG.normal(0, 1, 2000).tolist()

    assert numerical_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_an_identical_categorical_sample_does_not_move():
    values = ["a"] * 60 + ["b"] * 30 + ["c"] * 10

    assert categorical_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_a_sample_from_the_same_distribution_stays_below_the_stable_threshold():
    reference = RNG.normal(0, 1, 5000).tolist()
    current = RNG.normal(0, 1, 5000).tolist()

    assert numerical_index(reference, current) < 0.10


def test_a_shifted_distribution_is_reported_as_moved():
    reference = RNG.normal(0, 1, 5000).tolist()
    current = RNG.normal(1.5, 1, 5000).tolist()

    assert numerical_index(reference, current) > 0.25


def test_the_index_grows_with_the_size_of_the_shift():
    reference = RNG.normal(0, 1, 5000).tolist()

    small = numerical_index(reference, RNG.normal(0.3, 1, 5000).tolist())
    large = numerical_index(reference, RNG.normal(1.5, 1, 5000).tolist())

    assert small < large


def test_a_reversed_categorical_mix_is_reported_as_moved():
    reference = ["approve"] * 900 + ["decline"] * 100
    current = ["approve"] * 100 + ["decline"] * 900

    assert categorical_index(reference, current) > 0.25


def test_a_category_absent_from_one_side_does_not_divide_by_zero():
    """Clipping is what keeps a new category from producing an infinite index."""
    index = categorical_index(["a"] * 100, ["a"] * 50 + ["b"] * 50)

    assert np.isfinite(index)
    assert index > 0


def test_a_constant_column_reports_no_movement():
    """Its quantiles collapse onto one edge, which is not a distribution to compare."""
    assert numerical_index([5.0] * 100, [5.0] * 100) == 0.0


def test_a_constant_reference_against_a_varied_current_is_not_an_error():
    assert numerical_index([5.0] * 100, RNG.normal(0, 1, 100).tolist()) == 0.0


def test_an_empty_side_reports_no_movement_rather_than_failing():
    assert numerical_index([], [1.0, 2.0]) == 0.0
    assert numerical_index([1.0, 2.0], []) == 0.0
    assert categorical_index([], []) == 0.0


def test_buckets_come_from_the_reference():
    """Buckets that followed the new data would absorb the movement being measured."""
    reference = list(np.linspace(0, 1, 1000))
    current = list(np.linspace(10, 11, 1000))

    # Every current value falls in the last bucket of the reference's quantiles.
    assert numerical_index(reference, current) > 1.0


def test_comparison_returns_the_most_moved_first():
    reference = {"steady": RNG.normal(0, 1, 2000).tolist(),
                 "moved": RNG.normal(0, 1, 2000).tolist()}
    current = {"steady": RNG.normal(0, 1, 2000).tolist(),
               "moved": RNG.normal(2.0, 1, 2000).tolist()}

    signals = compare(reference, current, categorical=set())

    assert [signal.column for signal in signals] == ["moved", "steady"]
    assert signals[0].index > signals[1].index


def test_comparison_only_covers_columns_both_sides_carry():
    signals = compare({"a": [1.0], "b": [1.0]}, {"a": [1.0], "c": [1.0]}, set())

    assert [signal.column for signal in signals] == ["a"]


def test_a_column_named_categorical_is_treated_as_one():
    signals = compare({"x": ["a", "b"]}, {"x": ["a", "b"]}, categorical={"x"})

    assert signals[0].categorical is True


def test_severity_reads_from_the_conventional_thresholds():
    assert Signal("x", 0.05, False).severity == "stable"
    assert Signal("x", 0.10, False).severity == "moderate"
    assert Signal("x", 0.24, False).severity == "moderate"
    assert Signal("x", 0.25, False).severity == "significant"


def test_a_signal_reads_as_a_sentence():
    assert str(Signal("amount", 0.312, False)) == "amount: 0.312 (significant)"


def test_too_few_rows_is_reported_rather_than_dressed_up():
    assert insufficient({"a": [1.0] * 5}, minimum=50) is True
    assert insufficient({"a": [1.0] * 50}, minimum=50) is False
    assert insufficient({}, minimum=1) is True


def test_the_shortest_column_decides_sufficiency():
    assert insufficient({"a": [1.0] * 100, "b": [1.0] * 3}, minimum=50) is True
