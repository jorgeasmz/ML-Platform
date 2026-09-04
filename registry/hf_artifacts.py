"""An MLflow artifact backend stored in Hugging Face model repositories.

MLflow keeps runs, parameters and metrics in a database and artifacts wherever the
artifact root points. Every backend it ships with is either object storage that
bills or a filesystem that a free host does not keep across restarts, so the model
binaries would be the one part of the registry with nowhere durable to live.

Registering the `hf` scheme puts them in a Hugging Face model repository, which is
versioned by commit, free, and already where the published models of this portfolio
are served from. A URI is `hf://owner/repo`, with an optional path inside it and an
optional `?revision=` that pins reads to one commit.
"""

from __future__ import annotations

import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFolder
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository

SCHEME = "hf"
REPO_TYPE = "model"

# A commit is a 40 character hex digest. A branch or tag name is not, and only a
# branch can be written to.
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def parse_uri(uri: str) -> tuple[str, str, str | None]:
    """Splits an hf:// URI into repository, path inside it, and revision."""
    parsed = urlparse(uri)
    if parsed.scheme != SCHEME:
        raise MlflowException(f"Not a Hugging Face artifact URI: {uri}")

    owner = parsed.netloc
    segments = parsed.path.strip("/").split("/", 1)
    if not owner or not segments[0]:
        raise MlflowException(
            f"A Hugging Face artifact URI needs an owner and a repository: {uri}"
        )

    prefix = segments[1] if len(segments) > 1 else ""
    revision = parse_qs(parsed.query).get("revision", [None])[0]
    return f"{owner}/{segments[0]}", prefix, revision


class HuggingFaceArtifactRepository(ArtifactRepository):
    """Artifacts in one model repository, addressed by path within it."""

    def __init__(
        self,
        artifact_uri: str,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        self.repo_id, self.prefix, self.revision = parse_uri(artifact_uri)
        self.api = HfApi()

    def _remote(self, artifact_path: str | None) -> str:
        parts = [part for part in (self.prefix, artifact_path) if part]
        return posixpath.join(*parts) if parts else ""

    def _require_branch(self) -> None:
        """A pinned URI names one commit, and history is not rewritten to add to it."""
        if self.revision and COMMIT.match(self.revision):
            raise MlflowException(
                f"{self.artifact_uri} is pinned to a commit and cannot be written to. "
                "Write through the branch and pin the commit it returns."
            )

    def log_artifact(self, local_file, artifact_path=None) -> None:
        self._require_branch()
        name = Path(local_file).name
        target = self._remote(posixpath.join(artifact_path, name) if artifact_path else name)
        self.api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=target,
            repo_id=self.repo_id,
            repo_type=REPO_TYPE,
            revision=self.revision,
        )

    def log_artifacts(self, local_dir, artifact_path=None) -> None:
        self._require_branch()
        self.api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(local_dir),
            path_in_repo=self._remote(artifact_path) or None,
            repo_type=REPO_TYPE,
            revision=self.revision,
        )

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        remote = self._remote(path)
        try:
            entries = list(
                self.api.list_repo_tree(
                    self.repo_id,
                    remote or None,
                    revision=self.revision,
                    repo_type=REPO_TYPE,
                )
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            # MLflow reads an absent path as an empty directory rather than an error.
            return []

        listing = []
        for entry in entries:
            relative = entry.path
            if self.prefix:
                relative = posixpath.relpath(entry.path, self.prefix)
            is_dir = isinstance(entry, RepoFolder)
            listing.append(FileInfo(relative, is_dir, None if is_dir else entry.size))
        return sorted(listing, key=lambda info: info.path)

    def _download_file(self, remote_file_path, local_path) -> None:
        cached = hf_hub_download(
            repo_id=self.repo_id,
            filename=self._remote(remote_file_path),
            revision=self.revision,
            repo_type=REPO_TYPE,
        )
        # The hub owns the cached copy and hard-links it into place elsewhere, so it
        # is copied rather than moved.
        shutil.copyfile(cached, local_path)

    def delete_artifacts(self, artifact_path=None) -> None:
        self._require_branch()
        remote = self._remote(artifact_path)
        if not remote:
            raise MlflowException(
                "Refusing to delete every artifact in "
                f"{self.repo_id}. Name a path to delete."
            )
        self.api.delete_file(
            path_in_repo=remote,
            repo_id=self.repo_id,
            repo_type=REPO_TYPE,
            revision=self.revision,
        )
