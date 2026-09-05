import pytest
from mlflow import MlflowClient

from registry.gate import Measurement
from registry.models import get
from registry.promote import main, parse, run, summarise
from registry.store import production, promote, record

DATA = "d" * 16
REVISION = "c" * 40
ENVIRONMENT = {"python": "3.12.3", "packages": {}}

# The catalogue entry the tests drive, whose metric is a cost.
MODEL = "credit-risk"


@pytest.fixture
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def arguments(**overrides):
    argv = [
        "--model", overrides.get("model", MODEL),
        "--value", str(overrides.get("value", 310)),
        "--dataset", overrides.get("dataset", DATA),
        "--revision", overrides.get("revision", REVISION),
    ]
    if overrides.get("apply"):
        argv.append("--apply")
    return parse(argv)


def incumbent(client, value: float, dataset: str = DATA) -> None:
    model = get(MODEL)
    measurement = Measurement("", model.metric, value, dataset)
    version = record(client, model, "a" * 40, measurement, ENVIRONMENT)
    promote(client, model, version)


def test_the_first_candidate_is_promoted(client):
    decision = run(arguments(apply=True), client)

    assert decision.promote
    assert production(client, get(MODEL)).value == 310


def test_a_cheaper_candidate_replaces_the_incumbent(client):
    incumbent(client, 340)

    decision = run(arguments(value=310, apply=True), client)

    assert decision.promote
    assert production(client, get(MODEL)).value == 310


def test_a_costlier_candidate_leaves_production_alone(client):
    incumbent(client, 300)

    decision = run(arguments(value=340, apply=True), client)

    assert not decision.promote
    assert production(client, get(MODEL)).value == 300


def test_a_refused_candidate_is_still_registered_with_the_score_that_refused_it(client):
    incumbent(client, 300)

    run(arguments(value=340, apply=True), client)

    versions = client.search_model_versions(f"name='{MODEL}'")
    assert {float(version.tags["value"]) for version in versions} == {300.0, 340.0}


def test_without_apply_nothing_is_written(client):
    incumbent(client, 340)

    decision = run(arguments(value=310), client)

    assert decision.promote
    assert production(client, get(MODEL)).value == 340
    assert len(client.search_model_versions(f"name='{MODEL}'")) == 1


def test_a_candidate_on_other_data_is_held_and_production_is_untouched(client):
    incumbent(client, 340, dataset=DATA)

    decision = run(arguments(value=10, dataset="f" * 16, apply=True), client)

    assert not decision.promote
    assert "not comparable" in decision.reason
    assert production(client, get(MODEL)).value == 340


def test_an_unknown_model_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        parse(["--model", "no-such-model", "--value", "1", "--dataset", DATA,
               "--revision", REVISION])


def test_exit_status_carries_the_verdict(client, tmp_path, monkeypatch):
    """What lets a workflow step fail without understanding the metric."""
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)

    promoted = main(["--model", MODEL, "--value", "310", "--dataset", DATA,
                     "--revision", REVISION, "--apply"])
    held = main(["--model", MODEL, "--value", "400", "--dataset", DATA,
                 "--revision", "b" * 40, "--apply"])

    assert promoted == 0
    assert held == 1


def test_the_reason_reaches_the_workflow_summary(client, tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    summary.touch()
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    decision = run(arguments(apply=True), client)
    summarise(decision, MODEL)

    written = summary.read_text()
    assert "### Promoted: credit-risk" in written
    assert decision.reason in written


def test_no_summary_is_written_outside_a_workflow(monkeypatch, client):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    summarise(run(arguments(), client), MODEL)


@pytest.mark.parametrize("revision", ["", "not-a-commit", "77583b46"])
def test_a_revision_that_is_not_a_commit_is_refused(client, revision):
    """A failed shell substitution arrives as an empty string, not as an error."""
    with pytest.raises(ValueError, match="must be a 40 character commit"):
        run(arguments(revision=revision, apply=True), client)

    assert client.search_model_versions(f"name='{MODEL}'") == []
