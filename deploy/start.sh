#!/usr/bin/env bash
# Starts the tracking server with basic auth.
#
# MLflow reads its auth configuration with configparser and expands no environment
# variables in it, so the file is written here from the environment rather than
# committed with an administrator password in it.
set -euo pipefail

: "${MLFLOW_BACKEND_URI:?the backend store connection is required}"
: "${MLFLOW_ADMIN_USERNAME:?an administrator username is required}"
: "${MLFLOW_ADMIN_PASSWORD:?an administrator password is required}"
: "${MLFLOW_FLASK_SERVER_SECRET_KEY:?a stable secret key is required}"

# SQLAlchemy resolves a bare postgresql:// to psycopg2, which this project does not
# install. Without the driver in the URI the server starts and dies looking for it.
case "$MLFLOW_BACKEND_URI" in
    postgresql://*)
        echo "MLFLOW_BACKEND_URI must name the driver: postgresql+psycopg://" >&2
        exit 1
        ;;
esac

# The server runs CREATE TABLE and Alembic migrations at start-up, and a pooler in
# transaction mode may hand consecutive statements to different backends.
case "$MLFLOW_BACKEND_URI" in
    *-pooler.*)
        echo "MLFLOW_BACKEND_URI points at a pooled endpoint. Use the direct one:" >&2
        echo "the server migrates its own schema on start-up." >&2
        exit 1
        ;;
esac

config="${TMPDIR:-/tmp}/basic_auth.ini"
umask 077
cat > "$config" <<EOF
[mlflow]
default_permission = NO_PERMISSIONS
database_uri = ${MLFLOW_BACKEND_URI}
admin_username = ${MLFLOW_ADMIN_USERNAME}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF
export MLFLOW_AUTH_CONFIG_PATH="$config"

# The server starts a job subsystem with a process pool and two periodic tasks,
# online scoring and trace archival, neither of which this platform uses. Measured
# over the whole process tree: ten processes and 1,758 MB of PSS with it, two
# processes and 379 MB without. It is the difference between fitting the plan and
# being killed during start-up.
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

# The host header of every request is validated against this list, which defaults
# to localhost and private ranges. Without the public hostname the deployment
# answers its own health check and rejects everything else.
if [ -n "${RENDER_EXTERNAL_HOSTNAME:-}" ]; then
    export MLFLOW_SERVER_ALLOWED_HOSTS="${MLFLOW_SERVER_ALLOWED_HOSTS:-$RENDER_EXTERNAL_HOSTNAME}"
fi

# Artifacts are not proxied. They live in Hugging Face repositories and clients
# fetch them directly, so no model binary crosses this instance.
#
# One worker.
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_URI" \
    --no-serve-artifacts \
    --app-name basic-auth \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --workers 1
