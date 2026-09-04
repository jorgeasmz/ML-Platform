"""Whether a candidate replaces the model in production.

The comparison a gate exists to make is only meaningful when both numbers describe
the same thing. Two scores are comparable when they name the metric the model
promotes on and were measured on the same held-out data, so the gate establishes
that before it compares anything, and refuses rather than guesses when it cannot.

A refusal is not a failure of the candidate. A candidate measured on data the
incumbent was never scored against is unjudged, and the answer is to score the
incumbent on it rather than to promote on an incomparable pair.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from registry.models import RegisteredModel


@dataclass(frozen=True)
class Measurement:
    """One model version's score, and what makes it comparable to another's."""

    version: str
    metric: str
    value: float
    # Digest of the held-out data the score was measured on.
    dataset: str
    # The evaluation code that produced it, when the caller records one.
    commit: str | None = None


@dataclass(frozen=True)
class Decision:
    """The gate's answer, and the sentence that explains it."""

    promote: bool
    reason: str
    candidate: Measurement
    incumbent: Measurement | None = None

    def __str__(self) -> str:
        verdict = "promote" if self.promote else "hold"
        return f"{verdict}: {self.reason}"


def digest_bytes(data: bytes) -> str:
    """Identifies the held-out data a score was measured on.

    Every project that reports a score digests its held-out data with this, so two
    runs claiming the same data agree by construction rather than by convention.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def digest(path: str | Path) -> str:
    """The same, for held-out data that is a file."""
    return digest_bytes(Path(path).read_bytes())


def decide(
    model: RegisteredModel,
    candidate: Measurement,
    incumbent: Measurement | None,
) -> Decision:
    """Whether the candidate is promoted, and why."""
    def held(reason: str) -> Decision:
        return Decision(False, reason, candidate, incumbent)

    if candidate.metric != model.metric:
        return held(
            f"the candidate reports {candidate.metric} and {model.name} promotes on "
            f"{model.metric}"
        )

    if not model.clears_floor(candidate.value):
        return held(
            f"{candidate.metric} {candidate.value:g} does not clear the floor of "
            f"{model.floor:g}"
        )

    if incumbent is None:
        return Decision(
            True, "no version is in production, and the candidate clears the floor",
            candidate, None,
        )

    if incumbent.metric != model.metric:
        return held(
            f"the version in production reports {incumbent.metric}, which is not the "
            f"metric {model.name} promotes on"
        )

    if candidate.dataset != incumbent.dataset:
        return held(
            "the two were measured on different held-out data "
            f"({candidate.dataset} against {incumbent.dataset}), so their scores are "
            "not comparable. Score the version in production on the new data first"
        )

    direction = "above" if model.higher_is_better else "below"
    if not model.improves(candidate.value, incumbent.value):
        return held(
            f"{candidate.metric} {candidate.value:g} is not {direction} the "
            f"{incumbent.value:g} in production"
        )

    return Decision(
        True,
        f"{candidate.metric} {candidate.value:g} is {direction} the "
        f"{incumbent.value:g} in production, on the same held-out data",
        candidate,
        incumbent,
    )
