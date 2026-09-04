import pytest

from registry.models import CATALOGUE, RegisteredModel, get


def spec(**overrides) -> RegisteredModel:
    defaults = dict(
        name="example", repo="owner/repo", trained_by="https://example.invalid",
        metric="score", higher_is_better=True, packages=("numpy",),
    )
    return RegisteredModel(**(defaults | overrides))


def test_a_higher_is_better_metric_improves_upwards():
    model = spec(higher_is_better=True)

    assert model.improves(candidate=0.81, incumbent=0.80)
    assert not model.improves(candidate=0.80, incumbent=0.80)
    assert not model.improves(candidate=0.79, incumbent=0.80)


def test_a_cost_metric_improves_downwards():
    """Credit risk promotes on expected cost, where lower wins."""
    model = spec(higher_is_better=False)

    assert model.improves(candidate=310, incumbent=340)
    assert not model.improves(candidate=340, incumbent=340)


def test_a_tie_is_not_an_improvement():
    """An equal candidate is a redeploy with no measured reason behind it."""
    assert not spec(higher_is_better=True).improves(0.80, 0.80)
    assert not spec(higher_is_better=False).improves(0.80, 0.80)


def test_without_a_floor_every_candidate_clears_it():
    assert spec(floor=None).clears_floor(-1e9)


def test_a_floor_rejects_a_candidate_below_it():
    model = spec(higher_is_better=True, floor=0.70)

    assert model.clears_floor(0.71)
    assert model.clears_floor(0.70)
    assert not model.clears_floor(0.69)


def test_a_floor_on_a_cost_metric_rejects_a_candidate_above_it():
    model = spec(higher_is_better=False, floor=500)

    assert model.clears_floor(499)
    assert not model.clears_floor(501)


def test_an_artifact_uri_pins_to_a_revision_when_given():
    model = spec(repo="owner/repo")

    assert model.artifact_uri() == "hf://owner/repo"
    assert model.artifact_uri("abc123") == "hf://owner/repo?revision=abc123"


def test_an_unknown_model_names_the_known_ones():
    with pytest.raises(KeyError, match="credit-risk"):
        get("no-such-model")


def test_every_catalogue_entry_records_the_packages_it_marks_exact():
    """A package cannot be required to match exactly if its version is never recorded."""
    for model in CATALOGUE:
        assert model.exact <= set(model.packages), model.name


def test_catalogue_names_are_unique():
    names = [model.name for model in CATALOGUE]

    assert len(names) == len(set(names))
