"""The published page, built against a real registry backed by a temporary file."""

import pytest
from mlflow import MlflowClient

from registry.gate import Measurement
from registry.models import RegisteredModel
from registry.report import Entry, collect, render
from registry.store import promote, record

REVISION = "c" * 40
OTHER = "d" * 40
WRITTEN_UNDER = {"python": "3.12.3", "packages": {"scikit-learn": "1.9.0"}}


@pytest.fixture
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@pytest.fixture
def model() -> RegisteredModel:
    return RegisteredModel(
        name="credit-risk", repo="owner/repo", trained_by="https://example.invalid/Credit",
        metric="total_cost", higher_is_better=False, packages=("scikit-learn",),
        artifact_file="model.joblib",
    )


def registered(client, model, revision, value) -> str:
    return record(
        client, model, revision,
        Measurement("", model.metric, value, "d" * 16), WRITTEN_UNDER,
    )


def test_a_model_with_no_versions_collects_as_empty(client, model):
    assert collect(client, [model]) == {"credit-risk": []}


def test_versions_come_back_newest_first_with_the_serving_one_marked(client, model):
    first = registered(client, model, REVISION, 340)
    second = registered(client, model, OTHER, 310)
    promote(client, model, second)

    entries = collect(client, [model])["credit-risk"]

    assert [entry.version for entry in entries] == [second, first]
    assert [entry.serving for entry in entries] == [True, False]


def test_an_entry_carries_the_evidence_behind_it(client, model):
    version = registered(client, model, REVISION, 310)
    promote(client, model, version)

    (entry,) = collect(client, [model])["credit-risk"]

    assert entry.metric == "total_cost"
    assert entry.value == 310
    assert entry.dataset == "d" * 16
    assert entry.revision == REVISION
    assert entry.written_under == WRITTEN_UNDER


def test_the_page_is_self_contained(client, model):
    promote(client, model, registered(client, model, REVISION, 310))

    page = render(collect(client, [model]), [model])

    assert page.startswith("<!doctype html>")
    assert "<style>" in page
    # Nothing to fetch: no stylesheet, no script, no image.
    assert "<script" not in page
    assert "<link" not in page
    assert "<img" not in page


def test_the_page_names_the_serving_version_and_its_evidence(client, model):
    version = registered(client, model, REVISION, 310)
    promote(client, model, version)

    page = render(collect(client, [model]), [model])

    assert "production" in page
    assert "310" in page
    assert REVISION[:12] in page
    assert "scikit-learn 1.9.0" in page


def test_a_held_version_appears_without_the_production_mark(client, model):
    """The registry records what was tried, not only what shipped."""
    promote(client, model, registered(client, model, REVISION, 310))
    registered(client, model, OTHER, 400)

    page = render(collect(client, [model]), [model])

    assert page.count('<span class="badge">production</span>') == 1
    assert "400" in page


def test_the_direction_of_the_metric_is_stated(client, model):
    page = render(collect(client, [model]), [model])

    assert "lower is better" in page


def test_a_version_without_an_environment_record_says_so(client, model):
    version = record(
        client, model, REVISION,
        Measurement("", model.metric, 310, "d" * 16), {},
    )
    promote(client, model, version)

    page = render(collect(client, [model]), [model])

    assert "not recorded" in page


def test_an_unpinned_version_is_reported_as_such(client, model):
    """A version whose source names no commit identifies no bytes."""
    from registry.store import ensure_model

    ensure_model(client, model)
    version = client.create_model_version(model.name, "hf://owner/repo")
    promote(client, model, version.version)

    entries = collect(client, [model])["credit-risk"]

    assert entries[0].revision is None
    assert "unpinned" in render(collect(client, [model]), [model])


def test_the_page_escapes_what_it_prints():
    """Tags come from a registry a workflow writes, and are printed as text."""
    model = RegisteredModel(
        name="x", repo="owner/repo", trained_by="https://example.invalid",
        metric="score", higher_is_better=True, packages=(),
    )
    entries = {"x": [Entry(
        version="1", serving=False, metric="score", value=1.0,
        dataset="<script>alert(1)</script>", revision=None,
        written_under={"packages": {"<b>numpy</b>": "2.0"}},
    )]}

    page = render(entries, [model])

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;b&gt;numpy&lt;/b&gt;" in page
