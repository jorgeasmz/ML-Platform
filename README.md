# ML Platform

Operates the models the other projects in this portfolio train. It fits none of its
own: it decides which version of each is in production, refuses to serve one whose
artifact cannot be trusted to load, and blocks a promotion that is not an
improvement on the model it would replace.

![CI](https://github.com/jorgeasmz/ML-Platform/actions/workflows/ci.yml/badge.svg)

## Artifact storage

MLflow keeps runs, parameters and metrics in a database and artifacts wherever the
artifact root points. Every backend it ships with is either object storage that
bills or a filesystem that a free host does not keep across restarts, so the model
binaries are the one part of a registry with nowhere durable to live.

This project registers an `hf` URI scheme whose artifacts are Hugging Face model
repositories, which are versioned by commit, free, and already where the published
models of this portfolio are served from.

```
hf://owner/repo                    the repository root
hf://owner/repo/models/v3          a path inside it
hf://owner/repo?revision=<commit>  pinned to one commit, and read only
```

MLflow discovers the backend through the `mlflow.artifact_repository` entry point,
which is why the package is installed rather than merely importable. Without the
install, an `hf://` artifact root resolves to an unsupported scheme.

A URI carrying a commit refuses writes. A commit names bytes that already exist, and
the way to add to a pinned artifact is to write through the branch and pin the
commit that comes back.

## Environment record

A fitted estimator saved with pickle carries no record of what wrote it, and the
libraries that write it do not promise that a later version reads it correctly. The
failure is not an exception. An estimator can unpickle under a version that
reconstructs it differently and then predict, quietly, something else.

The version of every library a model declares is recorded beside its artifact when
it is registered, and checked before it is served.

| Difference | Behaviour |
|---|---|
| Major or minor moved | Load fails, naming every package and both versions |
| Patch moved, package declared exact | Load fails |
| Patch moved, package not declared exact | Reported, load proceeds |
| Package absent | Load fails |

Which packages are declared exact is a property of the model rather than a global
setting. `scikit-learn` documents no guarantee that a pickled estimator loads
correctly under another version, so the models that carry one declare it. Weights
loaded from a format that is versioned in itself do not need the same rule.

## Model catalogue

A registered model names the metric its promotion turns on, the direction that
improves that metric, and the libraries whose version has to match. Promoting on a
quantity other than the recorded one is not expressible.

| Model | Metric | Direction | Trained by |
|---|---|---|---|
| `credit-risk` | `total_cost` | lower | [Credit-Risk-Assessment](https://github.com/jorgeasmz/Credit-Risk-Assessment) |
| `irony` | `f1_ironic` | higher | [NLP-Sentiment-Analysis](https://github.com/jorgeasmz/NLP-Sentiment-Analysis) |

Credit risk promotes on the expected loss under the dataset's own cost matrix, where
a missed default costs five times a rejected good applicant. That project also
reports accuracy and ROC-AUC, and neither is what the decision is worth.

A tie is not an improvement. A candidate equal to the incumbent is a redeploy with
no measured reason behind it, so the gate declines it.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt   # installs this package, which registers the scheme

pytest              # 35 tests, offline
ruff check .
```

No test reaches the network or a database. The hub client is replaced by a double
that records what would have been uploaded and answers listings from a tree held in
the test, and the installed-version lookup is replaced by a mapping, with one test
that exercises the real lookup so the doubles cannot pass against code that never
worked.

## Project structure

```text
ML-Platform/
├── registry/
│   ├── hf_artifacts.py   # MLflow artifact backend over Hugging Face repositories
│   ├── environment.py    # What wrote an artifact, and whether this process may read it
│   └── models.py         # The catalogue, and the terms of a promotion
├── tests/                # pytest suite, offline
└── pyproject.toml        # The entry point MLflow discovers the backend through
```
