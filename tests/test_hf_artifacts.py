import pytest
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from registry.hf_artifacts import HuggingFaceArtifactRepository, parse_uri
from tests.conftest import repo_file, repo_folder

COMMIT = "a" * 40


def test_mlflow_resolves_the_scheme_to_this_backend():
    """The entry point is why an hf:// artifact root works at all."""
    resolved = get_artifact_repository("hf://jorgeasmz/fraud-stream-detector")

    assert isinstance(resolved, HuggingFaceArtifactRepository)


def test_a_uri_splits_into_repository_path_and_revision():
    assert parse_uri("hf://owner/repo") == ("owner/repo", "", None)
    assert parse_uri("hf://owner/repo/models/v3") == ("owner/repo", "models/v3", None)
    assert parse_uri(f"hf://owner/repo?revision={COMMIT}") == ("owner/repo", "", COMMIT)


def test_a_uri_of_another_scheme_is_refused():
    with pytest.raises(MlflowException, match="Not a Hugging Face artifact URI"):
        parse_uri("s3://owner/repo")


def test_a_uri_without_a_repository_is_refused():
    with pytest.raises(MlflowException, match="owner and a repository"):
        parse_uri("hf://owner")


def test_a_file_lands_under_the_uri_path(hub, tmp_path):
    local = tmp_path / "detector.joblib"
    local.write_bytes(b"fitted")
    repository = HuggingFaceArtifactRepository("hf://owner/repo/models")

    repository.log_artifact(str(local), artifact_path="v3")

    assert hub.uploads == [{
        "local": str(local), "remote": "models/v3/detector.joblib",
        "repo": "owner/repo", "type": "model", "revision": None,
    }]


def test_a_file_without_an_artifact_path_lands_at_the_root(hub, tmp_path):
    local = tmp_path / "detector.joblib"
    local.write_bytes(b"fitted")

    HuggingFaceArtifactRepository("hf://owner/repo").log_artifact(str(local))

    assert hub.uploads[0]["remote"] == "detector.joblib"


def test_a_directory_is_uploaded_as_a_folder(hub, tmp_path):
    (tmp_path / "run").mkdir()
    repository = HuggingFaceArtifactRepository("hf://owner/repo/models")

    repository.log_artifacts(str(tmp_path / "run"), artifact_path="v3")

    assert hub.folders[0]["remote"] == "models/v3"


def test_a_listing_is_relative_to_the_uri_path(hub):
    hub.tree["models"] = [
        repo_file("models/detector.joblib", 2048),
        repo_folder("models/v3"),
    ]
    repository = HuggingFaceArtifactRepository("hf://owner/repo/models")

    listing = repository.list_artifacts()

    assert [(info.path, info.is_dir, info.file_size) for info in listing] == [
        ("detector.joblib", False, 2048),
        ("v3", True, None),
    ]


def test_a_missing_path_lists_as_empty_rather_than_failing(hub):
    """MLflow reads an absent artifact path as an empty directory."""
    repository = HuggingFaceArtifactRepository("hf://owner/repo")

    assert repository.list_artifacts("never-written") == []


def test_a_download_copies_out_of_the_hub_cache(hub, monkeypatch, tmp_path):
    """The hub hard-links its cache elsewhere, so the file is copied rather than moved."""
    cached = tmp_path / "cached.joblib"
    cached.write_bytes(b"fitted")
    asked = {}

    def fake_download(*, repo_id, filename, revision, repo_type):
        asked.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(cached)

    monkeypatch.setattr("registry.hf_artifacts.hf_hub_download", fake_download)
    destination = tmp_path / "out.joblib"

    repository = HuggingFaceArtifactRepository(f"hf://owner/repo/models?revision={COMMIT}")
    repository._download_file("v3/detector.joblib", str(destination))

    assert destination.read_bytes() == b"fitted"
    assert cached.exists()
    assert asked == {
        "repo_id": "owner/repo", "filename": "models/v3/detector.joblib", "revision": COMMIT,
    }


def test_a_pinned_uri_refuses_to_be_written_to(hub, tmp_path):
    """A commit names bytes that already exist, and rewriting history is not an upload."""
    local = tmp_path / "detector.joblib"
    local.write_bytes(b"fitted")
    repository = HuggingFaceArtifactRepository(f"hf://owner/repo?revision={COMMIT}")

    with pytest.raises(MlflowException, match="pinned to a commit"):
        repository.log_artifact(str(local))

    assert hub.uploads == []


def test_a_branch_revision_is_writable(hub, tmp_path):
    local = tmp_path / "detector.joblib"
    local.write_bytes(b"fitted")

    repository = HuggingFaceArtifactRepository("hf://owner/repo?revision=candidate")
    repository.log_artifact(str(local))

    assert hub.uploads[0]["revision"] == "candidate"


def test_deleting_a_named_path_is_allowed(hub):
    HuggingFaceArtifactRepository("hf://owner/repo").delete_artifacts("v3/detector.joblib")

    assert hub.deletes[0]["remote"] == "v3/detector.joblib"


def test_deleting_everything_is_refused(hub):
    repository = HuggingFaceArtifactRepository("hf://owner/repo")

    with pytest.raises(MlflowException, match="Refusing to delete"):
        repository.delete_artifacts()

    assert hub.deletes == []
