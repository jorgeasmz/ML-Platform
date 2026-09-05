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

## Promotion gate

The comparison a gate exists to make is only meaningful when both numbers describe
the same thing, so the gate establishes that before it compares anything, and
refuses rather than guesses when it cannot.

| Condition | Outcome |
|---|---|
| The candidate reports a metric the model does not promote on | Held |
| The candidate does not clear the model's floor | Held |
| Nothing is in production | Promoted |
| The two were measured on different held-out data | Held, as incomparable |
| The candidate does not improve on the incumbent | Held |
| The candidate improves on the incumbent, on the same data | Promoted |

The fourth row is what separates a gate from a comparison of two numbers. A
candidate scored on data the incumbent was never scored against is unjudged, and
the answer is to score the incumbent on that data rather than to promote on an
incomparable pair. The refusal names the two digests that differed.

A tie is held. The floor is checked before the incumbent, so a candidate can beat a
bad incumbent and still not be worth serving.

```bash
python -m registry.promote --model credit-risk --value 312 \
    --dataset "$HOLDOUT_DIGEST" --revision "$ARTIFACT_COMMIT" --apply
```

Exit status is the answer: 0 promotes, 1 holds. That is what lets a workflow step
fail on a candidate that is not an improvement without the workflow having to
understand the metric or read the registry. The reason is also written to the run
summary. Without `--apply` the gate decides and writes nothing, which is what a
pull request wants.

## Registry

A version records where its artifact is, the score that justified it, the held-out
data that score was measured on, and the library versions it was written under.
Those travel as tags on the version rather than only on the run that produced it,
so asking what is in production and on what evidence is one lookup.

Production is an alias rather than a stage. MLflow 3 replaced stages with aliases,
and an alias is the better fit regardless: it points at exactly one version, and
moving it is the whole of a promotion.

A version's source pins the artifact commit rather than naming the branch, so the
version identifies the bytes that were scored and not whatever the branch points at
afterwards.

The candidate is registered whichever way the gate decides. A refused version stays
in the registry carrying the score that refused it, which is the record of what was
tried.

## Resolving what to serve

A service that builds its own model serves whatever the last build produced, which
nothing measured and nothing compared. The resolver answers the opposite question:
which version does the registry say is in production, and may this process read it.

```bash
python -m registry.resolve --model credit-risk --into ./model
```

It reads the version behind the `production` alias, checks the recorded library
versions against the ones present, and only then downloads. The order matters: a
service that fetches the artifact and discovers afterwards that it cannot read it
has already paid for the fetch.

The download goes through MLflow's artifact repository, which resolves the
version's `hf://` source through the backend registered here. The registration path
and the serving path reach the same bytes through the same code, and the source is
pinned to the commit that was scored rather than to a branch that has moved since.

Nothing in production is an error rather than an empty result. A service has no
model to fall back to, and a candidate has to pass the gate before anything can
serve it.

## Deployment

| Component | Host | |
|---|---|---|
| Tracking server and registry | Render | [ml-platform-registry.onrender.com](https://ml-platform-registry.onrender.com), credentialed |
| Runs, versions and auth users | Neon, PostgreSQL | |
| Artifacts | Hugging Face | one model repository per model |

No model binary crosses the tracking server. It runs with `--no-serve-artifacts`,
so a client resolves an `hf://` source and fetches from Hugging Face directly, and
the instance carries metadata only.

### Access

The server requires a credential on every route but `/health`, which is exempt and
is what the platform's own health check reaches. Measured against the deployment,
`/health` answers 200 unauthenticated while `/` and the registry API answer 401
with `WWW-Authenticate: Basic realm="mlflow"`.

It is not published with a browse credential. MLflow's basic auth gates reading and
editing a resource, but creation is ungated: with workspaces disabled, which is the
default, `_can_create_in_workspace` returns true for any authenticated user
whatever their `default_permission`. A published read-only credential would
therefore also be a credential that can create registered models.

MLflow reads its auth configuration with `configparser` and expands no environment
variables in it, so `deploy/start.sh` writes the file at start-up from the
environment rather than committing an administrator password to the repository.

What is published instead is a static page built from the registry by a workflow
that holds the credential. It carries which version of each model is serving, the
measurement that promoted it, the commit its artifact lives at, and the library
versions it was written under, in one self-contained file with no stylesheet, no
script and nothing that can be written to.

The page rebuilds daily as well as on a change to this repository, because
promotions happen in the repositories that train the models rather than in this
one, and a page rebuilt only on pushes here would report yesterday's production.

### Connection

`MLFLOW_BACKEND_URI` names the driver and points at the direct endpoint:

```
postgresql+psycopg://user:password@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require
```

Both halves of that are checked at start-up, because both fail late and unhelpfully
otherwise. SQLAlchemy resolves a bare `postgresql://` to psycopg2, which this
project does not install, so the server would start and die looking for it. And the
server runs `CREATE TABLE` and Alembic migrations on start-up, which a pooler in
transaction mode may spread across different backends; the connection count that
would justify a pooler does not arise at one worker.

### Sizing

The server starts a job subsystem with a process pool and two periodic tasks,
online scoring and trace archival. This platform uses neither, and the pool is
what decides whether the instance fits.

| `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` | Processes over 20 MB | Summed PSS |
|---|---:|---:|
| `true`, the default | 10 | 1,758 MB |
| **`false`** | **2** | **379 MB** |

Measured over the whole process tree, since the pool is forked and neither its
processes nor their memory appear in the parent.

379 MB against the 512 MB the plan allows is modest headroom rather than
comfortable, and it is what the free tier permits. One worker is what ships: a
second would put the pair past the limit.

The figure to measure is the tree, not the process. A count taken from the
processes whose command line matches the server misses everything forked from it,
and reports 195 MB for a deployment that is killed during start-up.

The free plan sleeps an idle instance, so the first request after a quiet period
pays the start-up. Nothing on the critical path of a deployment depends on the
server answering quickly.

## Serving on Kubernetes

`deploy/chart` is one chart parameterised by model and image rather than a set of
manifests per service, so a second model is a values file rather than a copy.

The two probes ask different questions, because the service answers them
differently on purpose. Its root endpoint returns 200 even when the model failed
to load, which is what keeps a bad artifact from becoming a restart loop, so
liveness reads the status and readiness reads the body and asserts `model_loaded`.
That is the same check the service's own compose file makes, stated once more
where Kubernetes can see it.

Requests and limits are set at 256Mi and 512Mi, which is the bound the service is
already known to run within on the free tier it is deployed to.

### What the cluster job proves

`helm lint` and a rendered template establish that the YAML is valid, which a
manifest that crash-loops also is. So the workflow builds the real image from the
service's own Dockerfile, loads it into a `kind` cluster, installs the chart, and
then asks the cluster questions that a rendering cannot answer.

| Question | How it is answered |
|---|---|
| Does the pod start | `kubectl rollout status`, which fails on a crash loop |
| Does readiness pass | The rollout does not complete until it does |
| Does the Service route | A port-forward and a request through it |
| Which model is being served | The response carries the artifact hash |
| Can the autoscaler read the workload | metrics-server is installed and the HPA is polled for a utilisation |

The last two are the ones worth having. Measured on a run: the service reports
`model_version` `4d17fdeb3b1d`, the same artifact hash the registry recorded when
the gate scored it, and the autoscaler reports `cpu: 29%/70%` rather than
`<unknown>`, which is the difference between an autoscaler that exists and one
that reads its target.

### Behaviour under change and under load

Starting is not operating. Two more questions are asked of the same cluster, and
both fail the job when the answer is wrong.

| Property | Measured |
|---|---|
| The autoscaler adds a replica under load | 1 to 2 replicas after 50 s, at 110% of the 70% target |
| A rolling update drops nothing | 14,614 requests through a full pod replacement, 0 failed |

The second was also measured with the drain pause disabled, and dropped nothing
either: on one node the endpoint removal propagates before it matters. The pause
is kept because the window it covers widens with the number of nodes the change
has to reach, which is the case a single-node cluster cannot produce. Reporting it
as the reason the test passes would be reporting something the measurement does
not show.

### What no cluster does

No cluster serves public traffic, and none is planned. There is no managed
Kubernetes on a free tier without a card; the one provider whose free tier
includes it had no Ampere capacity in the region, and its terms reclaim an idle
instance after seven days below 20% utilisation, which is the exact profile of a
portfolio demo nobody visits. A dead URL is worse than an absent one.

What is lost with it is a public address served from Kubernetes and anything
about running a cluster over weeks: node upgrades, resource pressure, certificate
rotation. What is kept is every property above, asserted on every change to the
manifests.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt   # installs this package, which registers the scheme

pytest              # 116 tests, offline
ruff check .
```

No test reaches the network. The hub client is replaced by a double that records
what would have been uploaded and answers listings from a tree held in the test,
and the installed-version lookup is replaced by a mapping, with one test that
exercises the real lookup so the doubles cannot pass against code that never worked.

The registry and the gate command run against a real MLflow registry backed by a
temporary SQLite file rather than against a double, so an API that moves under the
project fails here rather than in deployment.

## Project structure

```text
ML-Platform/
├── registry/
│   ├── hf_artifacts.py   # MLflow artifact backend over Hugging Face repositories
│   ├── environment.py    # What wrote an artifact, and whether this process may read it
│   ├── models.py         # The catalogue, and the terms of a promotion
│   ├── gate.py           # Whether a candidate replaces the model in production
│   ├── store.py          # Model versions, their evidence, and the production alias
│   ├── promote.py        # The command a training pipeline calls, and its exit status
│   ├── resolve.py        # The production artifact on local disk, or why it is not
│   └── report.py         # The published page, built from the registry
├── deploy/start.sh       # Writes the auth configuration, then runs the server
├── deploy/chart/         # One chart, parameterised by model and image
├── render.yaml           # The tracking server, on the free plan
├── tests/                # pytest suite, offline
└── pyproject.toml        # The entry point MLflow discovers the backend through
```
