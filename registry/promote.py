"""The command a training pipeline calls to ask whether its candidate ships.

Exit status is the answer: 0 promotes, 1 holds. That is what lets a workflow step
fail on a candidate that is not an improvement, without the workflow having to
understand the metric or read the registry itself.

The candidate is registered whichever way the gate decides, so a refused version
stays in the registry with the score that refused it. What promotion changes is the
alias, and nothing else.

Usage:
    python -m registry.promote --model credit-risk --value 312 \\
        --dataset <digest> --revision <artifact commit> [--apply]
"""

from __future__ import annotations

import argparse
import os
import sys

from mlflow import MlflowClient

from registry.environment import capture
from registry.gate import Decision, Measurement, decide
from registry.models import BY_NAME, get
from registry.store import production, promote, record


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(BY_NAME))
    parser.add_argument("--value", required=True, type=float, help="the candidate's score")
    parser.add_argument(
        "--dataset", required=True,
        help="digest of the held-out data the score was measured on",
    )
    parser.add_argument(
        "--revision", required=True,
        help="commit of the artifact in its Hugging Face repository",
    )
    parser.add_argument("--commit", default=None, help="the evaluation code that scored it")
    parser.add_argument("--run-id", default=None, help="the MLflow run that produced it")
    parser.add_argument(
        "--apply", action="store_true",
        help="register the candidate and move the alias when the gate passes",
    )
    return parser.parse_args(argv)


def run(arguments: argparse.Namespace, client: MlflowClient) -> Decision:
    model = get(arguments.model)
    candidate = Measurement(
        version=arguments.revision,
        metric=model.metric,
        value=arguments.value,
        dataset=arguments.dataset,
        commit=arguments.commit,
    )

    decision = decide(model, candidate, production(client, model))
    if not arguments.apply:
        return decision

    version = record(
        client, model, arguments.revision, candidate,
        capture(model.packages), arguments.run_id,
    )
    if decision.promote:
        promote(client, model, version)
    return decision


def summarise(decision: Decision, model: str) -> None:
    """Puts the reason in the workflow summary, where a reader looks first."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    verdict = "Promoted" if decision.promote else "Held"
    with open(path, "a", encoding="utf-8") as summary:
        summary.write(f"### {verdict}: {model}\n\n{decision.reason}\n\n")


def main(argv: list[str] | None = None) -> int:
    arguments = parse(argv)
    client = MlflowClient()

    decision = run(arguments, client)
    print(decision)
    summarise(decision, arguments.model)
    return 0 if decision.promote else 1


if __name__ == "__main__":
    sys.exit(main())
