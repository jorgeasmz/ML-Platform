"""The models this platform operates, and what decides whether a candidate replaces one.

A registered model names the metric its promotion turns on, the direction that
improves that metric, and the libraries whose version has to match for its artifact
to be loadable at all. The gate reads this and nothing else, so promoting on a
quantity other than the one recorded here is not expressible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegisteredModel:
    """One model, its artifact repository, and the terms of its promotion."""

    name: str
    repo: str
    trained_by: str
    metric: str
    higher_is_better: bool
    # Libraries whose version is recorded beside the artifact.
    packages: tuple[str, ...]
    # Of those, the ones whose version must match exactly for a load to be trusted.
    # A library that gives no cross-version guarantee for its serialised estimators
    # belongs here; one whose format is stable across patches does not.
    exact: frozenset[str] = field(default_factory=frozenset)
    # An absolute bar a candidate must clear regardless of the incumbent. None means
    # the incumbent is the only bar.
    floor: float | None = None

    def improves(self, candidate: float, incumbent: float) -> bool:
        """Whether the candidate is better than the incumbent on this model's metric."""
        return candidate > incumbent if self.higher_is_better else candidate < incumbent

    def clears_floor(self, candidate: float) -> bool:
        if self.floor is None:
            return True
        return self.improves(candidate, self.floor) or candidate == self.floor

    def artifact_uri(self, revision: str | None = None) -> str:
        """Where the artifact lives, optionally pinned to one commit."""
        uri = f"hf://{self.repo}"
        return f"{uri}?revision={revision}" if revision else uri


SKLEARN_STACK = ("scikit-learn", "numpy", "scipy", "joblib")
TRANSFORMERS_STACK = ("transformers", "torch", "tokenizers", "numpy")

CATALOGUE = (
    RegisteredModel(
        name="credit-risk",
        repo="jorgeasmz/credit-risk-scorer",
        trained_by="https://github.com/jorgeasmz/Credit-Risk-Assessment",
        # The dataset's own cost matrix, where a missed default costs five times a
        # rejected good applicant. Accuracy and ROC-AUC are reported by that project
        # and neither is what the decision is worth.
        metric="total_cost",
        higher_is_better=False,
        packages=SKLEARN_STACK,
        # scikit-learn documents no guarantee that a pickled estimator loads
        # correctly under another version, and the failure is silent.
        exact=frozenset({"scikit-learn"}),
    ),
    RegisteredModel(
        name="irony",
        repo="jorgeasmz/distilbert-irony-tweeteval",
        trained_by="https://github.com/jorgeasmz/NLP-Sentiment-Analysis",
        metric="f1_ironic",
        higher_is_better=True,
        packages=TRANSFORMERS_STACK,
        # Weights are loaded from a serialisation format that is versioned in itself
        # rather than reconstructed from a pickle, so a patch difference is reported
        # rather than blocking.
        exact=frozenset(),
    ),
)

BY_NAME = {model.name: model for model in CATALOGUE}


def get(name: str) -> RegisteredModel:
    try:
        return BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(BY_NAME))
        raise KeyError(f"'{name}' is not a registered model. Known: {known}") from None
