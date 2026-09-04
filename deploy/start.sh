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

# Artifacts are not proxied. They live in Hugging Face repositories and clients
# fetch them directly, so no model binary crosses this instance.
#
# One worker. Measured at 256 MB resident and flat under load, against the 512 MB
# the plan allows; a second worker would put the pair close enough to the limit to
# be decided by chance.
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_URI" \
    --no-serve-artifacts \
    --app-name basic-auth \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --workers 1
