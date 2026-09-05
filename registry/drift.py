"""Whether the data a model is scoring still looks like the data it was fitted on.

A model is fitted on one population and then asked about whatever arrives. Nothing
in the serving path notices when those stop being the same, and the accuracy that
justified the promotion was measured on the first one.

The measure here is the population stability index, which is the symmetric
Kullback-Leibler divergence between two discretised distributions. It is the
standard for this in credit scoring, where it comes from, and its thresholds are
conventional rather than derived: below 0.1 is treated as no meaningful shift,
0.1 to 0.25 as moderate, above 0.25 as significant.

Evidently would compute this and more. It is not used here because it requires
plotly below 6 and one of the projects it would monitor pins plotly 7, so
installing it would mean downgrading the dashboard of the service being watched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONE = 0.10
MODERATE = 0.25

# Neither distribution may have an empty bucket, since the index divides by one
# and takes the logarithm of the ratio. This is the conventional floor.
FLOOR = 1e-4

BINS = 10


@dataclass(frozen=True)
class Signal:
    """One column, and how far its distribution has moved."""

    column: str
    index: float
    categorical: bool

    @property
    def severity(self) -> str:
        if self.index < NONE:
            return "stable"
        return "moderate" if self.index < MODERATE else "significant"

    def __str__(self) -> str:
        return f"{self.column}: {self.index:.3f} ({self.severity})"


def _index(reference: np.ndarray, current: np.ndarray) -> float:
    """Population stability between two sets of proportions over the same buckets."""
    reference = np.clip(reference, FLOOR, None)
    current = np.clip(current, FLOOR, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def categorical_index(reference: list, current: list) -> float:
    """Proportions per category, over the categories either side contains."""
    categories = sorted(set(reference) | set(current))
    if not categories:
        return 0.0

    def proportions(values: list) -> np.ndarray:
        total = len(values) or 1
        counts = {category: 0 for category in categories}
        for value in values:
            counts[value] += 1
        return np.array([counts[c] / total for c in categories])

    return _index(proportions(reference), proportions(current))


def numerical_index(reference: list, current: list, bins: int = BINS) -> float:
    """Proportions per quantile bucket of the reference, which is what defines them.

    The buckets come from the reference rather than from the combined sample: the
    question is how far the new data has moved from the old, and buckets that
    shift with the new data would absorb the very movement being measured.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0

    quantiles = np.quantile(reference, np.linspace(0, 1, bins + 1))
    # A column with few distinct values collapses its quantiles onto each other.
    edges = np.unique(quantiles)
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    def proportions(values: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=edges)
        return counts / (counts.sum() or 1)

    return _index(proportions(reference), proportions(current))


def compare(
    reference: dict[str, list],
    current: dict[str, list],
    categorical: set[str],
) -> list[Signal]:
    """Every column both sides carry, most moved first."""
    signals = []
    for column in sorted(set(reference) & set(current)):
        is_categorical = column in categorical
        index = (
            categorical_index(reference[column], current[column])
            if is_categorical
            else numerical_index(reference[column], current[column])
        )
        signals.append(Signal(column, index, is_categorical))
    return sorted(signals, key=lambda signal: signal.index, reverse=True)


def insufficient(current: dict[str, list], minimum: int) -> bool:
    """Whether there is enough scored data to say anything at all.

    A monitor with nothing to compare reports that, rather than a small number of
    rows dressed up as a distribution.
    """
    if not current:
        return True
    return min(len(values) for values in current.values()) < minimum
