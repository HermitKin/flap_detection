#!/usr/bin/env bash
set -euo pipefail

container_name="${LEVEL_GAUGE_CONTAINER:-yolo_container}"
container_workdir="/workspace/level_gauge_batch"

docker exec -it \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env USER="${USER:-appuser}" \
  --env LOGNAME="${LOGNAME:-${USER:-appuser}}" \
  --env MPLCONFIGDIR=/tmp/level_gauge_matplotlib \
  --env YOLO_CONFIG_DIR=/tmp/level_gauge_ultralytics \
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/level_gauge_torch \
  --workdir "${container_workdir}" \
  "${container_name}" \
  python remote_batch_reading.py "$@"
