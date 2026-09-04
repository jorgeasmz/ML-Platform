import pytest
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile, RepoFolder


class FakeHub:
    """Records what would have been sent to the hub, and answers listings from a tree.

    Every test in the default run goes through this. Nothing reaches the network.
    """

    def __init__(self, tree: dict[str, list] | None = None) -> None:
        self.tree = tree if tree is not None else {}
        self.uploads: list[dict] = []
        self.folders: list[dict] = []
        self.deletes: list[dict] = []

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, revision):
        self.uploads.append({
            "local": str(path_or_fileobj), "remote": path_in_repo,
            "repo": repo_id, "type": repo_type, "revision": revision,
        })

    def upload_folder(self, *, repo_id, folder_path, path_in_repo, repo_type, revision):
        self.folders.append({
            "local": str(folder_path), "remote": path_in_repo,
            "repo": repo_id, "type": repo_type, "revision": revision,
        })

    def delete_file(self, *, path_in_repo, repo_id, repo_type, revision):
        self.deletes.append({"remote": path_in_repo, "repo": repo_id, "revision": revision})

    def list_repo_tree(self, repo_id, path_in_repo=None, *, revision=None, repo_type=None,
                       recursive=False, expand=False):
        key = path_in_repo or ""
        if key not in self.tree:
            raise EntryNotFoundError(f"no such path: {key}")
        return list(self.tree[key])


def repo_file(path: str, size: int) -> RepoFile:
    return RepoFile(path=path, size=size, oid="0" * 8)


def repo_folder(path: str) -> RepoFolder:
    return RepoFolder(path=path, oid="0" * 8, tree_id="t")


@pytest.fixture
def hub(monkeypatch):
    """Replaces the hub client the repository builds in its constructor."""
    fake = FakeHub()
    monkeypatch.setattr("registry.hf_artifacts.HfApi", lambda *a, **k: fake)
    return fake
