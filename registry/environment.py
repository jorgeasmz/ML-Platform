"""What produced an artifact, and whether the process loading it can.

A fitted estimator saved with pickle carries no record of what wrote it, and the
libraries that write it do not promise that a later version reads it correctly.
The failure is not an exception: an estimator can unpickle under a version that
reconstructs it differently and then predict, quietly, something else.

So the version of every library that matters is recorded beside the artifact when
it is registered, and checked before it is served. A mismatch that can change a
prediction stops the load; one that cannot is reported and does not.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

MISSING = "absent"

RECORD = "environment.json"


@dataclass(frozen=True)
class Difference:
    """One package whose version is not what the artifact was written under."""

    package: str
    recorded: str
    present: str
    blocking: bool

    def __str__(self) -> str:
        return f"{self.package}: recorded {self.recorded}, present {self.present}"


def installed(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return MISSING


def capture(packages: tuple[str, ...]) -> dict:
    """The versions an artifact is being written under."""
    return {
        "python": platform.python_version(),
        "packages": {name: installed(name) for name in sorted(packages)},
    }


def _series(value: str) -> str:
    """Major and minor, which is the granularity at which a format usually moves."""
    return ".".join(value.split(".")[:2])


def differences(record: dict, exact: frozenset[str] = frozenset()) -> list[Difference]:
    """Every package whose present version departs from the recorded one.

    A package named in `exact` blocks on any difference at all, which is what the
    libraries that give no cross-version guarantee for their pickles require.
    Everything else blocks only when major or minor moved.
    """
    found = []
    for name, recorded in sorted(record.get("packages", {}).items()):
        present = installed(name)
        if present == recorded:
            continue

        blocking = present == MISSING or (
            name in exact or _series(present) != _series(recorded)
        )
        found.append(Difference(name, recorded, present, blocking))
    return found


def assert_loadable(record: dict, exact: frozenset[str] = frozenset()) -> list[Difference]:
    """Raises when a difference could change a prediction. Returns the rest."""
    found = differences(record, exact)
    blocking = [difference for difference in found if difference.blocking]
    if blocking:
        raise EnvironmentMismatch(blocking)
    return found


class EnvironmentMismatch(RuntimeError):
    """The artifact was written under versions this process cannot be trusted to read."""

    def __init__(self, blocking: list[Difference]) -> None:
        self.blocking = blocking
        listed = "; ".join(str(difference) for difference in blocking)
        super().__init__(
            f"The artifact was written under different versions of {listed}. "
            "Serving it here would predict from an estimator reconstructed by code "
            "that never fitted it."
        )
