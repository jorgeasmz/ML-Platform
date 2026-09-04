"""The registry layer, against a real MLflow registry backed by a temporary SQLite file.

These exercise MLflow itself rather than a double, so an API that moves under the
project fails here. Nothing reaches the network.
"""

import pytest
from mlflow import MlflowClient

from registry.gate import Measurement
from registry.models import RegisteredModel
from registry.store import (
    PRODUCTION,
    ensure_model,
    environment_of,
    production,
    promote,
    record,
)

REVISION = "c" * 40
ENVIRONMENT = {"python": "3.12.3", "packages": {"scikit-learn": "1.9.0"}}


@pytest.fixture
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@pytest.fixture
def model() -> RegisteredModel:
    return RegisteredModel(
        name="example", repo="owner/repo", trained_by="https://example.invalid",
        metric="total_cost", higher_is_better=False, packages=("scikit-learn",),
    )


def measured(value: float, dataset: str = "d" * 16) -> Measurement:
    return Measurement(version="", metric="total_cost", value=value, dataset=dataset)


def test_nothing_is_in_production_before_anything_is_registered(client, model):
    assert production(client, model) is None


def test_a_registered_version_is_not_yet_in_production(client, model):
    record(client, model, REVISION, measured(340), ENVIRONMENT)

    assert production(client, model) is None


def test_promotion_makes_the_version_readable_as_production(client, model):
    version = record(client, model, REVISION, measured(340), ENVIRONMENT)

    promote(client, model, version)

    serving = production(client, model)
    assert serving.value == 340
    assert serving.metric == "total_cost"
    assert serving.version == version


def test_the_source_pins_the_artifact_commit(client, model):
    """A version names the bytes that were scored, not what the branch points at later."""
    version = record(client, model, REVISION, measured(340), ENVIRONMENT)

    stored = client.get_model_version(model.name, version)

    assert stored.source == f"hf://owner/repo?revision={REVISION}"


def test_the_environment_record_round_trips(client, model):
    version = record(client, model, REVISION, measured(340), ENVIRONMENT)

    assert environment_of(client, model, version) == ENVIRONMENT


def test_a_version_without_an_environment_record_reads_as_empty(client, model):
    ensure_model(client, model)
    version = client.create_model_version(model.name, "hf://owner/repo").version

    assert environment_of(client, model, version) == {}


def test_a_version_carrying_no_measurement_reads_as_none(client, model):
    """A version registered outside this platform has no score to compare against."""
    ensure_model(client, model)
    version = client.create_model_version(model.name, "hf://owner/repo").version
    client.set_registered_model_alias(model.name, PRODUCTION, version)

    assert production(client, model) is None


def test_promoting_again_moves_the_alias(client, model):
    first = record(client, model, REVISION, measured(340), ENVIRONMENT)
    second = record(client, model, "d" * 40, measured(310), ENVIRONMENT)
    promote(client, model, first)

    promote(client, model, second)

    assert production(client, model).value == 310


def test_registering_twice_does_not_fail_on_the_registered_model(client, model):
    record(client, model, REVISION, measured(340), ENVIRONMENT)
    record(client, model, "e" * 40, measured(330), ENVIRONMENT)

    assert len(client.search_model_versions(f"name='{model.name}'")) == 2


def test_a_float_survives_the_tag_round_trip(client, model):
    """Tags are text, and a rounded score would change what the gate compares."""
    version = record(client, model, REVISION, measured(0.6931471805599453), ENVIRONMENT)
    promote(client, model, version)

    assert production(client, model).value == 0.6931471805599453
