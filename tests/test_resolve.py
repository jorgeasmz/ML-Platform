"""Resolving the production artifact, against a real registry and a fake hub."""

import pytest
from mlflow import MlflowClient

from registry import resolve as resolve_module
from registry.environment import EnvironmentMismatch
from registry.gate import Measurement
from registry.models import RegisteredModel
from registry.resolve import NothingInProduction, resolve
from registry.store import promote, record

REVISION = "c" * 40
WRITTEN_UNDER = {"python": "3.12.3", "packages": {"scikit-learn": "1.9.0", "numpy": "2.5.2"}}


@pytest.fixture
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@pytest.fixture
def model(monkeypatch) -> RegisteredModel:
    """A catalogue of one, so the tests do not turn on the real entries."""
    spec = RegisteredModel(
        name="example", repo="owner/repo", trained_by="https://example.invalid",
        metric="total_cost", higher_is_better=False,
        packages=("scikit-learn", "numpy"), exact=frozenset({"scikit-learn"}),
        artifact_file="model.joblib",
    )
    monkeypatch.setattr(resolve_module, "BY_NAME", {spec.name: spec})
    monkeypatch.setattr(resolve_module, "get", lambda name: spec)
    return spec


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Stands in for the hub, recording what the version's source asked for."""
    asked = []

    class FakeRepository:
        def __init__(self, source):
            self.source = source

        def download_artifacts(self, artifact_path, destination):
            asked.append({"source": self.source, "path": artifact_path,
                          "into": destination})
            target = tmp_path / "downloaded"
            target.write_bytes(b"fitted")
            return str(target)

    monkeypatch.setattr(resolve_module, "get_artifact_repository", FakeRepository)
    return asked


def present(monkeypatch, versions: dict[str, str]) -> None:
    monkeypatch.setattr(
        "registry.environment.installed",
        lambda name: versions.get(name, "absent"),
    )


def registered(client, model, environment=None, value=340) -> str:
    measurement = Measurement("", model.metric, value, "d" * 16)
    # An empty record is a real case and is falsy, so it is distinguished by identity.
    written = WRITTEN_UNDER if environment is None else environment
    return record(client, model, REVISION, measurement, written)


def test_nothing_in_production_is_reported_as_such(client, model, downloads):
    with pytest.raises(NothingInProduction, match="pass the gate"):
        resolve(model.name, "/tmp/unused", client)


def test_a_promoted_version_resolves_to_a_local_path(
    client, model, downloads, monkeypatch, tmp_path
):
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.2"})
    promote(client, model, registered(client, model))

    resolved = resolve(model.name, tmp_path / "into", client)

    assert resolved.model == "example"
    assert resolved.path.read_bytes() == b"fitted"
    assert resolved.measurement.value == 340


def test_the_download_uses_the_version_source_pinned_to_its_commit(
    client, model, downloads, monkeypatch, tmp_path
):
    """The bytes fetched are the bytes that were scored, not the branch tip."""
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.2"})
    promote(client, model, registered(client, model))

    resolve(model.name, tmp_path / "into", client)

    assert downloads[0]["source"] == f"hf://owner/repo?revision={REVISION}"
    assert downloads[0]["path"] == "model.joblib"


def test_a_model_without_a_named_file_downloads_the_whole_repository(
    client, model, downloads, monkeypatch, tmp_path
):
    """Weights split across files are the repository, not one artifact."""
    monkeypatch.setattr(resolve_module, "get", lambda name: RegisteredModel(
        name=model.name, repo=model.repo, trained_by=model.trained_by,
        metric=model.metric, higher_is_better=False, packages=model.packages,
    ))
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.2"})
    promote(client, model, registered(client, model))

    resolve(model.name, tmp_path / "into", client)

    assert downloads[0]["path"] == ""


def test_a_blocking_version_difference_refuses_the_load(
    client, model, downloads, monkeypatch, tmp_path
):
    present(monkeypatch, {"scikit-learn": "1.8.2", "numpy": "2.5.2"})
    promote(client, model, registered(client, model))

    with pytest.raises(EnvironmentMismatch, match="scikit-learn"):
        resolve(model.name, tmp_path / "into", client)


def test_the_refusal_happens_before_anything_is_downloaded(
    client, model, downloads, monkeypatch, tmp_path
):
    """A service that fetches and then refuses has already paid for the fetch."""
    present(monkeypatch, {"scikit-learn": "1.8.2", "numpy": "2.5.2"})
    promote(client, model, registered(client, model))

    with pytest.raises(EnvironmentMismatch):
        resolve(model.name, tmp_path / "into", client)

    assert downloads == []


def test_a_tolerated_difference_is_reported_and_does_not_refuse(
    client, model, downloads, monkeypatch, tmp_path
):
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.9"})
    promote(client, model, registered(client, model))

    resolved = resolve(model.name, tmp_path / "into", client)

    assert [d.package for d in resolved.tolerated] == ["numpy"]
    assert all(not d.blocking for d in resolved.tolerated)


def test_a_version_registered_without_an_environment_record_still_resolves(
    client, model, downloads, monkeypatch, tmp_path
):
    """Nothing recorded is nothing to contradict, and it is reported as no difference."""
    present(monkeypatch, {"scikit-learn": "1.8.2"})
    promote(client, model, registered(client, model, environment={}))

    resolved = resolve(model.name, tmp_path / "into", client)

    assert resolved.tolerated == []


def test_the_command_prints_the_version_and_its_measurement(
    client, model, downloads, monkeypatch, tmp_path, capsys
):
    """What a build step reads to know which model it just baked in."""
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.9"})
    version = registered(client, model)
    promote(client, model, version)

    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)

    exit_code = resolve_module.main(["--model", model.name, "--into", str(tmp_path / "into")])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert f"example version {version}" in printed
    assert "total_cost 340" in printed
    assert "tolerated: numpy: recorded 2.5.2, present 2.5.9" in printed


def test_the_command_refuses_a_model_outside_the_catalogue(model):
    with pytest.raises(SystemExit):
        resolve_module.main(["--model", "no-such-model", "--into", "/tmp/unused"])


def test_a_pointer_carries_only_what_a_build_may_know(
    client, model, downloads, monkeypatch, tmp_path
):
    """A serving image needs the bytes, not a credential to the registry."""
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    version = registered(client, model)
    promote(client, model, version)

    pointer = resolve_module.describe(model.name, client)

    assert pointer["repo"] == "owner/repo"
    assert pointer["revision"] == REVISION
    assert pointer["artifact_file"] == "model.joblib"
    assert pointer["version"] == version
    assert pointer["metric"] == "total_cost"
    assert pointer["value"] == 340
    assert pointer["environment"] == WRITTEN_UNDER
    assert pointer["url"] == (
        f"https://huggingface.co/owner/repo/resolve/{REVISION}/model.joblib"
    )


def test_describing_downloads_nothing(client, model, downloads, tmp_path):
    promote(client, model, registered(client, model))

    resolve_module.describe(model.name, client)

    assert downloads == []


def test_a_pointer_without_a_named_file_carries_no_url(
    client, model, downloads, monkeypatch, tmp_path
):
    """A repository of weights is not one URL, so none is offered."""
    monkeypatch.setattr(resolve_module, "get", lambda name: RegisteredModel(
        name=model.name, repo=model.repo, trained_by=model.trained_by,
        metric=model.metric, higher_is_better=False, packages=model.packages,
    ))
    promote(client, model, registered(client, model))

    pointer = resolve_module.describe(model.name, client)

    assert "url" not in pointer
    assert pointer["artifact_file"] is None


def test_the_command_writes_the_pointer_as_json(
    client, model, downloads, monkeypatch, tmp_path, capsys
):
    import json

    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    promote(client, model, registered(client, model))

    exit_code = resolve_module.main(["--model", model.name, "--describe"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["revision"] == REVISION
    assert downloads == []


def test_downloading_without_a_destination_is_refused(model):
    with pytest.raises(SystemExit):
        resolve_module.main(["--model", model.name])
