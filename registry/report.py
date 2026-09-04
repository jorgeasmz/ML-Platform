"""A page saying which version of each model is serving, and on what evidence.

The tracking server is not public. Its basic auth gates reading a resource but not
creating one, so a published browse credential would also be a credential that can
create registered models. What is published instead is this: a static page built
from the registry by a workflow that holds the credential, carrying what a reader
needs and nothing that can be written to.

Usage: python -m registry.report --into site
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException, RestException

from registry.models import CATALOGUE, RegisteredModel
from registry.store import PRODUCTION, RECORD, measurement_of


@dataclass(frozen=True)
class Entry:
    """One registered version, and whether it is the one serving."""

    version: str
    serving: bool
    metric: str | None
    value: float | None
    dataset: str | None
    revision: str | None
    written_under: dict = field(default_factory=dict)


def serving_version(client: MlflowClient, model: RegisteredModel) -> str | None:
    try:
        return client.get_model_version_by_alias(model.name, PRODUCTION).version
    except (MlflowException, RestException):
        return None


def collect(client: MlflowClient, models=CATALOGUE) -> dict[str, list[Entry]]:
    """Every version of every model, newest first, with the serving one marked.

    The alias is read separately because a version listing does not carry it: the
    search returns entities whose aliases are empty however the alias is set.
    """
    collected: dict[str, list[Entry]] = {}
    for model in models:
        serving = serving_version(client, model)
        try:
            versions = client.search_model_versions(f"name='{model.name}'")
        except (MlflowException, RestException):
            versions = []

        entries = []
        for version in sorted(versions, key=lambda v: int(v.version), reverse=True):
            measured = measurement_of(version)
            recorded = version.tags.get(RECORD)
            entries.append(Entry(
                version=version.version,
                serving=version.version == serving,
                metric=measured.metric if measured else None,
                value=measured.value if measured else None,
                dataset=measured.dataset if measured else None,
                revision=version.source.partition("?revision=")[2] or None,
                written_under=json.loads(recorded) if recorded else {},
            ))
        collected[model.name] = entries
    return collected


STYLE = """
:root { color-scheme: dark; }
body { background: #0d1117; color: #c9d1d9; font: 15px/1.6 ui-sans-serif, system-ui, sans-serif;
       margin: 0; padding: 3rem 1.5rem; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; color: #f0f6fc; }
h2 { font-size: 1.15rem; margin: 2.6rem 0 .3rem; color: #f0f6fc; }
p.lede, p.note { color: #8b949e; margin: .3rem 0 1rem; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0 0; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #21262d;
         vertical-align: top; }
th { color: #8b949e; font-weight: 600; font-size: .82rem; text-transform: uppercase;
     letter-spacing: .04em; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, SFMono-Regular, monospace; font-size: .88rem; }
tr.serving td { background: #0f2417; }
.badge { background: #1f6f3f; color: #d6ffe4; border-radius: 999px; padding: .1rem .55rem;
         font-size: .74rem; font-weight: 600; }
.held { color: #8b949e; }
a { color: #58a6ff; }
footer { margin-top: 3rem; color: #6e7681; font-size: .85rem; }
"""


def _cell(value) -> str:
    return html.escape("" if value is None else str(value))


def _environment(entry: Entry) -> str:
    packages = entry.written_under.get("packages", {})
    if not packages:
        return '<span class="held">not recorded</span>'
    return ", ".join(
        f"{html.escape(name)} {html.escape(version)}"
        for name, version in sorted(packages.items())
    )


def render(collected: dict[str, list[Entry]], models=CATALOGUE) -> str:
    """One self-contained page. No stylesheet, no script, nothing to fetch."""
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>ML Platform registry</title>",
        f"<style>{STYLE}</style></head><body><main>",
        "<h1>ML Platform registry</h1>",
        '<p class="lede">Which version of each model is in production, and the '
        "measurement that promoted it. A candidate is promoted only when it improves "
        "on the version it would replace, measured on the same held-out data.</p>",
    ]

    by_name = {model.name: model for model in models}
    for name, entries in collected.items():
        model = by_name[name]
        direction = "higher is better" if model.higher_is_better else "lower is better"
        parts.append(f"<h2>{html.escape(name)}</h2>")
        source = html.escape(model.trained_by)
        project = html.escape(model.trained_by.rsplit("/", 1)[-1])
        repo = html.escape(model.repo)
        parts.append(
            f'<p class="note">Promoted on <code>{html.escape(model.metric)}</code>, '
            f'{direction}. Trained by <a href="{source}">{project}</a>, '
            f'artifact in <a href="https://huggingface.co/{repo}">{repo}</a>.</p>'
        )

        if not entries:
            parts.append('<p class="held">No version has been registered.</p>')
            continue

        parts.append(
            "<table><thead><tr><th>Version</th><th>Metric</th>"
            "<th>Held-out data</th><th>Artifact</th><th>Written under</th>"
            "</tr></thead><tbody>"
        )
        for entry in entries:
            row = ' class="serving"' if entry.serving else ""
            badge = ' <span class="badge">production</span>' if entry.serving else ""
            value = "" if entry.value is None else f"{entry.value:g}"
            artifact = (
                f'<a class="mono" href="https://huggingface.co/{html.escape(model.repo)}'
                f'/tree/{_cell(entry.revision)}">{_cell(entry.revision[:12])}</a>'
                if entry.revision else '<span class="held">unpinned</span>'
            )
            parts.append(
                f"<tr{row}><td>{_cell(entry.version)}{badge}</td>"
                f'<td class="n">{_cell(value)}</td>'
                f'<td class="mono">{_cell(entry.dataset)}</td>'
                f"<td>{artifact}</td>"
                f'<td class="mono">{_environment(entry)}</td></tr>'
            )
        parts.append("</tbody></table>")

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<footer>Generated {generated} from the registry. A version not marked "
        "production was registered and held: the gate records what was tried, not "
        "only what shipped.</footer></main></body></html>"
    )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", default="site", help="directory to write index.html into")
    arguments = parser.parse_args(argv)

    destination = Path(arguments.into)
    destination.mkdir(parents=True, exist_ok=True)
    page = destination / "index.html"
    page.write_text(render(collect(MlflowClient())), encoding="utf-8")

    print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
