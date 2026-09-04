"""The artifact a service should serve, and the check that it can.

A service that builds its own model serves whatever the last build produced, which
nothing measured and nothing compared. This resolves the opposite: the version the
registry says is in production, identified by the commit that was scored, fetched
from where that commit lives, and refused if the process cannot be trusted to read
it.

The download goes through MLflow's own artifact repository, which resolves the
version's `hf://` source through the backend this package registers. The serving
path and the registration path therefore use the same code to reach the same bytes.

Usage: python -m registry.resolve --model credit-risk --into ./model
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException, RestException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from registry.environment import Difference, assert_loadable
from registry.gate import Measurement
from registry.hf_artifacts import parse_uri
from registry.models import BY_NAME, RegisteredModel, get
from registry.store import PRODUCTION, environment_of, measurement_of


class NothingInProduction(LookupError):
    """No version carries the production alias, so there is nothing to serve."""


@dataclass(frozen=True)
class Resolved:
    """What was fetched, which version it is, and what it was measured at."""

    model: str
    version: str
    source: str
    path: Path
    measurement: Measurement | None
    # Version differences that do not block a load but are worth reporting.
    tolerated: list[Difference]


def production_version(client: MlflowClient, model: RegisteredModel):
    try:
        return client.get_model_version_by_alias(model.name, PRODUCTION)
    except (MlflowException, RestException) as error:
        raise NothingInProduction(
            f"No version of '{model.name}' carries the {PRODUCTION} alias. "
            "A candidate has to pass the gate before anything can serve it."
        ) from error


def describe(model_name: str, client: MlflowClient | None = None) -> dict:
    """The production version as a pointer, without fetching anything.

    A serving image needs the bytes but not the registry: it has no business
    holding a credential to it. This is what a build reads instead, and every
    field in it is public, so the image downloads over plain HTTPS.
    """
    model = get(model_name)
    client = client or MlflowClient()

    version = production_version(client, model)
    _, _, revision = parse_uri(version.source)
    measured = measurement_of(version)

    pointer = {
        "model": model.name,
        "version": version.version,
        "repo": model.repo,
        "revision": revision,
        "artifact_file": model.artifact_file,
        "environment": environment_of(client, model, version.version),
    }
    if measured:
        pointer |= {
            "metric": measured.metric,
            "value": measured.value,
            "dataset": measured.dataset,
        }
    if model.artifact_file and revision:
        pointer["url"] = (
            f"https://huggingface.co/{model.repo}/resolve/{revision}/{model.artifact_file}"
        )
    return pointer


def fetch(source: str, artifact_file: str | None, destination: Path) -> Path:
    """Downloads the artifact named by a version's source into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    repository = get_artifact_repository(source)
    return Path(
        repository.download_artifacts(artifact_file or "", str(destination))
    )


def resolve(
    model_name: str,
    destination: str | Path,
    client: MlflowClient | None = None,
) -> Resolved:
    """The production artifact on local disk, or an exception saying why not."""
    model = get(model_name)
    client = client or MlflowClient()

    version = production_version(client, model)
    recorded = environment_of(client, model, version.version)

    # Before the bytes are fetched, not after: a service that downloads and then
    # discovers it cannot read them has already paid for the download.
    tolerated = assert_loadable(recorded, model.exact)

    path = fetch(version.source, model.artifact_file, Path(destination))
    return Resolved(
        model=model.name,
        version=version.version,
        source=version.source,
        path=path,
        measurement=measurement_of(version),
        tolerated=tolerated,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(BY_NAME))
    parser.add_argument("--into", help="directory to download into")
    parser.add_argument(
        "--describe", action="store_true",
        help="write the pointer as JSON and download nothing",
    )
    arguments = parser.parse_args(argv)

    if arguments.describe:
        print(json.dumps(describe(arguments.model), indent=2, sort_keys=True))
        return 0

    if not arguments.into:
        parser.error("--into is required unless --describe is given")

    resolved = resolve(arguments.model, arguments.into)

    measured = resolved.measurement
    if measured:
        print(f"{resolved.model} version {resolved.version}: "
              f"{measured.metric} {measured.value:g} on held-out data {measured.dataset}")
    else:
        print(f"{resolved.model} version {resolved.version}, carrying no measurement")

    for difference in resolved.tolerated:
        print(f"  tolerated: {difference}")

    print(resolved.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
