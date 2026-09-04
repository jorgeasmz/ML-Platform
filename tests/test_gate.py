from registry.gate import Measurement, decide, digest
from registry.models import RegisteredModel

DATA = "0" * 16
OTHER_DATA = "1" * 16


def spec(**overrides) -> RegisteredModel:
    defaults = dict(
        name="example", repo="owner/repo", trained_by="https://example.invalid",
        metric="score", higher_is_better=True, packages=("numpy",),
    )
    return RegisteredModel(**(defaults | overrides))


def measured(value: float, metric: str = "score", dataset: str = DATA) -> Measurement:
    return Measurement(version="abc123", metric=metric, value=value, dataset=dataset)


def test_a_better_candidate_on_the_same_data_is_promoted():
    decision = decide(spec(), measured(0.81), measured(0.80))

    assert decision.promote
    assert "on the same held-out data" in decision.reason


def test_a_worse_candidate_is_held():
    decision = decide(spec(), measured(0.79), measured(0.80))

    assert not decision.promote
    assert "is not above the 0.8 in production" in decision.reason


def test_an_equal_candidate_is_held():
    """A redeploy with no measured reason behind it."""
    assert not decide(spec(), measured(0.80), measured(0.80)).promote


def test_a_cost_metric_promotes_the_lower_candidate():
    model = spec(higher_is_better=False, metric="total_cost")
    better = measured(310, metric="total_cost")
    worse = measured(340, metric="total_cost")

    assert decide(model, better, worse).promote
    assert not decide(model, worse, better).promote


def test_the_first_version_is_promoted_with_nothing_to_beat():
    decision = decide(spec(), measured(0.42), None)

    assert decision.promote
    assert "no version is in production" in decision.reason


def test_the_first_version_still_has_to_clear_a_floor():
    decision = decide(spec(floor=0.70), measured(0.42), None)

    assert not decision.promote
    assert "does not clear the floor" in decision.reason


def test_a_candidate_below_the_floor_is_held_even_when_it_beats_production():
    """A model can improve on a bad incumbent and still not be worth serving."""
    decision = decide(spec(floor=0.70), measured(0.65), measured(0.60))

    assert not decision.promote
    assert "floor" in decision.reason


def test_a_candidate_reporting_another_metric_is_held():
    decision = decide(spec(metric="score"), measured(0.99, metric="accuracy"), measured(0.80))

    assert not decision.promote
    assert "promotes on score" in decision.reason


def test_an_incumbent_recorded_under_another_metric_is_held():
    decision = decide(spec(), measured(0.90), measured(0.80, metric="accuracy"))

    assert not decision.promote
    assert "not the metric" in decision.reason


def test_scores_on_different_data_are_refused_rather_than_compared():
    """The check that makes this a gate rather than a comparison of two numbers."""
    decision = decide(spec(), measured(0.99, dataset=OTHER_DATA), measured(0.80))

    assert not decision.promote
    assert "not comparable" in decision.reason
    assert "Score the version in production on the new data first" in decision.reason


def test_the_data_check_runs_before_the_comparison():
    """A candidate that looks worse on other data is still refused as incomparable."""
    decision = decide(spec(), measured(0.10, dataset=OTHER_DATA), measured(0.80))

    assert "not comparable" in decision.reason


def test_the_metric_check_runs_before_the_floor():
    model = spec(metric="score", floor=0.70)
    decision = decide(model, measured(0.10, metric="accuracy"), None)

    assert "promotes on score" in decision.reason


def test_a_decision_reads_as_a_sentence():
    assert str(decide(spec(), measured(0.81), measured(0.80))).startswith("promote: ")
    assert str(decide(spec(), measured(0.79), measured(0.80))).startswith("hold: ")


def test_the_same_bytes_digest_the_same_and_different_bytes_do_not(tmp_path):
    first = tmp_path / "holdout.csv"
    second = tmp_path / "copy.csv"
    changed = tmp_path / "changed.csv"
    first.write_bytes(b"a,b\n1,2\n")
    second.write_bytes(b"a,b\n1,2\n")
    changed.write_bytes(b"a,b\n1,3\n")

    assert digest(first) == digest(second)
    assert digest(first) != digest(changed)
    assert len(digest(first)) == 16
