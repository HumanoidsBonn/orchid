#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${ORCHID_SERVER_ROOT:-/home/orchid/orchid_server}"
ENV_FILE="${ORCHID_MANIFEST_ENV_FILE:-${REPO_ROOT}/.env.manifest}"

source /opt/ros/humble/setup.bash

if [[ -f /home/orchid/hsr/install/setup.bash ]]; then
  source /home/orchid/hsr/install/setup.bash
fi

if [[ -f /home/orchid/ros2_ws/install/setup.bash ]]; then
  source /home/orchid/ros2_ws/install/setup.bash
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${REPO_ROOT}"
exec python3 -m src.orchestrator.dispatch_server "$@"
