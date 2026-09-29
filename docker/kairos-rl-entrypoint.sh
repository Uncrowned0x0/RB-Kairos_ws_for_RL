#!/bin/bash
# =============================================================================
# kairos-rl-entrypoint.sh — Entrypoint for Kairos WS RL container
# ROS 2 Jazzy + Gazebo Harmonic + PyTorch + Stable-Baselines3
# =============================================================================

set -eo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/robot/ros2_ws}"
SRC_DIR="${SRC_DIR:-${WORKSPACE_DIR}/src}"
CONTAINER_USERNAME="${CONTAINER_USERNAME:-robot}"

log() {
  echo -e "\033[0;34m[kairos-rl-entrypoint]\033[0m $*"
}

source_setup_file() {
  local setup_file="$1"
  if [ -f "${setup_file}" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
  fi
}

prepare_workspace_permissions() {
  mkdir -p "${WORKSPACE_DIR}" "${SRC_DIR}" \
           "${WORKSPACE_DIR}/build" \
           "${WORKSPACE_DIR}/install" \
           "${WORKSPACE_DIR}/log" \
           "${WORKSPACE_DIR}/checkpoints" \
           "${WORKSPACE_DIR}/rl_level0" \
           "${WORKSPACE_DIR}/tb_logs"

  CONTAINER_GROUP="$(id -gn "${CONTAINER_USERNAME}" 2>/dev/null || echo "${CONTAINER_USERNAME}")"
  chown -R "${CONTAINER_USERNAME}:${CONTAINER_GROUP}" \
           "${WORKSPACE_DIR}/build" \
           "${WORKSPACE_DIR}/install" \
           "${WORKSPACE_DIR}/log" \
           "${WORKSPACE_DIR}/checkpoints" \
           "${WORKSPACE_DIR}/rl_level0" \
           "${WORKSPACE_DIR}/tb_logs" 2>/dev/null || true
}

clean_stale_ipc() {
  # Clean residual FastRTPS shared memory segments to avoid mutex lockups
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
}

main() {
  source_setup_file "/opt/ros/${ROS_DISTRO}/setup.bash"
  source_setup_file "${WORKSPACE_DIR}/install/setup.bash"

  if [ "$(id -u)" = "0" ]; then
    clean_stale_ipc
    prepare_workspace_permissions

    if [ "$#" -eq 0 ]; then
      log "RL container ready (sleeping). Use './kairos_rl.sh shell' to interact."
      exec gosu "${CONTAINER_USERNAME}" sleep infinity
    fi

    exec gosu "${CONTAINER_USERNAME}" "$@"
  else
    # Direct execution as non-root user (no gosu to avoid EPERM)
    if [ "$#" -eq 0 ]; then
      log "RL container ready (sleeping). Use './kairos_rl.sh shell' to interact."
      exec sleep infinity
    fi

    exec "$@"
  fi
}

main "$@"
