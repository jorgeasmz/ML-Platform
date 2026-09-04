"""Model versions in the MLflow registry, and which of them is in production.

A version records where its artifact is, the score that justified it, the held-out
data that score was measured on, and the library versions it was written under.
Those travel as tags on the version rather than only on the run that produced it,
so asking what is in production and on what evidence is one lookup.

Production is an alias rather than a stage. MLflow 3 replaced stages with aliases,
and an alias is the better fit regardless: it points at exactly one version and
moving it is the whole of a promotion.
"""

from __future__ import annotations

import json

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException, RestException

from registry.environment import RECORD
from registry.gate import Measurement
from registry.models import RegisteredModel

PRODUCTION = "production"

METRIC = "metric"
VALUE = "value"
DATASET = "dataset"
COMMIT = "commit"


def ensure_model(client: MlflowClient, model: RegisteredModel) -> None:
    """Creates the registered model if this is the first version of it."""
    try:
        client.get_registered_model(model.name)
    except (MlflowException, RestException):
        client.create_registered_model(
            model.name,
            description=f"Promotes on {model.metric}. Trained by {model.trained_by}.",
        )


def _measurement(version) -> Measurement | None:
    """Reads a measurement off a version's tags, or None when it carries none."""
    tags = version.tags
    if METRIC not in tags or VALUE not in tags:
        return None
    return Measurement(
        version=version.version,
        metric=tags[METRIC],
        value=float(tags[VALUE]),
        dataset=tags.get(DATASET, ""),
        commit=tags.get(COMMIT) or None,
    )


def production(client: MlflowClient, model: RegisteredModel) -> Measurement | None:
    """The measurement behind the version currently serving, if there is one."""
    try:
        version = client.get_model_version_by_alias(model.name, PRODUCTION)
    except (MlflowException, RestException):
        return None
    return _measurement(version)


def record(
    client: MlflowClient,
    model: RegisteredModel,
    revision: str,
    measurement: Measurement,
    environment: dict,
    run_id: str | None = None,
) -> str:
    """Registers a candidate version pinned to the artifact commit. Returns its number.

    The source is pinned rather than left on the branch, so the version names the
    bytes that were scored and not whatever the branch points at later.
    """
    ensure_model(client, model)
    version = client.create_model_version(
        name=model.name,
        source=model.artifact_uri(revision),
        run_id=run_id,
        tags={
            METRIC: measurement.metric,
            VALUE: repr(measurement.value),
            DATASET: measurement.dataset,
            COMMIT: measurement.commit or "",
            RECORD: json.dumps(environment, sort_keys=True),
        },
    )
    return version.version


def environment_of(client: MlflowClient, model: RegisteredModel, version: str) -> dict:
    """The library versions one model version's artifact was written under."""
    stored = client.get_model_version(model.name, version).tags.get(RECORD)
    return json.loads(stored) if stored else {}


def promote(client: MlflowClient, model: RegisteredModel, version: str) -> None:
    """Moves the production alias, which is the whole of a promotion."""
    client.set_registered_model_alias(model.name, PRODUCTION, version)
