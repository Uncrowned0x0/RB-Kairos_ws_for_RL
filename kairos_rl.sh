#!/usr/bin/env bash
# ==============================================================================
# Author: Kamil BENMADI
# Email: kamil.benmadi@sigma-clermont.fr
# GitHub: https://github.com/Uncrowned0x0
# ==============================================================================
# =============================================================================
# kairos_rl.sh — Convenience script for Kairos WS RL workspace
# Robotnik KAIROS+ & UR5e Mobile Manipulation — Reinforcement Learning
#
# Usage:
#   ./kairos_rl.sh <command> [options]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="kairos_rl_sim"
ROS2_WS="/home/robot/ros2_ws"

# Terminal colors
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
MAGENTA=$'\033[0;35m'
BOLD=$'\033[1m'
NC=$'\033[0m' # No Color

info()    { echo -e "${BLUE}[kairos-rl]${NC} $*" >&2; }
success() { echo -e "${GREEN}[kairos-rl]${NC} $*" >&2; }
warn()    { echo -e "${YELLOW}[kairos-rl]${NC} $*" >&2; }
error()   { echo -e "${RED}[kairos-rl]${NC} $*" >&2; exit 1; }

# Detect if currently running inside the Docker container
is_inside_container() {
    [ -f /.dockerenv ] || [ -f /run/.containerenv ]
}

# Verify Docker installation and daemon accessibility
check_docker() {
    if is_inside_container; then
        return 0
    fi
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker Engine."
    fi
    if ! docker info &> /dev/null; then
        error "Docker daemon is not accessible. Try: sudo usermod -aG docker \$USER && newgrp docker"
    fi
}

# Automatically detect NVIDIA GPU acceleration
get_compose_files() {
    local compose_args=("-f" "${SCRIPT_DIR}/docker-compose.yaml")
    if [[ "${FORCE_CPU:-0}" == "1" ]]; then
        info "CPU mode forced by user."
    elif command -v nvidia-smi &> /dev/null && docker info 2>/dev/null | grep -iq "nvidia"; then
        compose_args+=("-f" "${SCRIPT_DIR}/docker-compose.gpu.yaml")
    elif command -v nvidia-smi &> /dev/null; then
        if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q "nvidia"; then
            compose_args+=("-f" "${SCRIPT_DIR}/docker-compose.gpu.yaml")
        fi
    fi
    echo "${compose_args[@]}"
}

# Verify that the container is running (auto-start if stopped)
require_running() {
    if is_inside_container; then
        return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "Container '${CONTAINER_NAME}' is not running. Starting container..."
        COMPOSE_ARGS=($(get_compose_files))
        cd "${SCRIPT_DIR}"
        if [ -n "${DISPLAY:-}" ] && command -v xhost &>/dev/null; then
            xhost +local:docker >/dev/null 2>&1 || true
        fi
        LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose "${COMPOSE_ARGS[@]}" up -d
        sleep 2
    fi
}

# Execute a command inside the container as 'robot' user with ROS environment sourced
docker_exec() {
    if is_inside_container; then
        cd "${ROS2_WS}"
        set +u
        source /opt/ros/jazzy/setup.bash
        if [ -f "${ROS2_WS}/install/setup.bash" ]; then
            source "${ROS2_WS}/install/setup.bash"
        fi
        set -u
        eval "$*"
    else
        local it_flag=("-i")
        if [ -t 0 ] && [ -t 1 ]; then
            it_flag=("-it")
        fi
        docker exec "${it_flag[@]}" -u robot "${CONTAINER_NAME}" bash -c "cd ${ROS2_WS} && set +u && source /opt/ros/jazzy/setup.bash && if [ -f install/setup.bash ]; then source install/setup.bash; fi && set -u && $*"
    fi
}

# =============================================================================
CMD="${1:-help}"
shift || true

case "${CMD}" in

    # ── BUILD ────────────────────────────────────────────────────────────────
    build)
        if is_inside_container; then
            warn "You are currently running inside the Docker container. Build images from the host system."
            exit 0
        fi
        check_docker
        info "Building Kairos RL Docker image..."
        info "Detecting hardware acceleration..."
        COMPOSE_ARGS=($(get_compose_files))
        cd "${SCRIPT_DIR}"
        LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose "${COMPOSE_ARGS[@]}" build "$@"
        success "Image built successfully."
        info "Next steps: start container with './kairos_rl.sh start' then compile with './kairos_rl.sh colcon_build'."
        ;;

    # ── START ────────────────────────────────────────────────────────────────
    start)
        if is_inside_container; then
            info "You are already inside the running container."
            exit 0
        fi
        check_docker
        info "Starting container '${CONTAINER_NAME}'..."
        COMPOSE_ARGS=($(get_compose_files))
        cd "${SCRIPT_DIR}"
        # Authorize X11 GUI access if display server is reachable
        if [ -n "${DISPLAY:-}" ] && command -v xhost &>/dev/null; then
            xhost +local:docker >/dev/null 2>&1 || true
        fi
        LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose "${COMPOSE_ARGS[@]}" up -d
        success "Container started in background."
        info "To open an interactive shell: ./kairos_rl.sh shell"
        ;;

    # ── STOP ─────────────────────────────────────────────────────────────────
    stop)
        if is_inside_container; then
            warn "You are inside the container. Exit first ('exit'), then run './kairos_rl.sh stop' on host."
            exit 0
        fi
        check_docker
        info "Stopping container..."
        cd "${SCRIPT_DIR}"
        docker compose down
        success "Container stopped."
        ;;

    # ── RESTART ──────────────────────────────────────────────────────────────
    restart)
        if is_inside_container; then
            warn "To restart the container, exit first ('exit'), then run './kairos_rl.sh restart' on host."
            exit 0
        fi
        check_docker
        info "Restarting container..."
        cd "${SCRIPT_DIR}"
        docker compose down
        COMPOSE_ARGS=($(get_compose_files))
        LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose "${COMPOSE_ARGS[@]}" up -d
        success "Container restarted."
        ;;

    # ── SHELL ────────────────────────────────────────────────────────────────
    shell)
        if is_inside_container; then
            info "You are already in an interactive container shell (user: $(whoami))."
            exit 0
        fi
        check_docker
        require_running
        info "Opening interactive shell inside container (user: robot)..."
        docker exec -it -u robot "${CONTAINER_NAME}" bash
        ;;

    # ── COLCON BUILD ─────────────────────────────────────────────────────────
    colcon|colcon_build|colcon-build|build_ws|build-ws|compile)
        if [ "${1:-}" = "build" ]; then
            shift || true
        fi
        check_docker
        require_running
        info "Compiling ROS 2 workspace inside container..."
        docker_exec "colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release $*"
        success "Workspace built successfully."
        ;;

    # ── TRAIN CURRICULUM (Recommended) ───────────────────────────────────────
    train|train_curriculum)
        check_docker
        require_running
        info "Launching KAIROS Curriculum Learning (4 Levels)..."
        info "Passed arguments: $*"
        if [ "$#" -eq 0 ]; then
            warn "No arguments provided. Launching Level 0 (Reach) by default with 2 parallel envs and 500k timesteps."
            info "Recommended usage: ./kairos_rl.sh train --level 0 --num-envs 4 --total-timesteps 500000"
            docker_exec "ros2 run kairos_rl train_curriculum --level 0 --num-envs 2 --total-timesteps 500000"
        else
            docker_exec "ros2 run kairos_rl train_curriculum $*"
        fi
        ;;

    # ── TRAIN PPO STANDARD (Single instance) ─────────────────────────────────
    train_ppo)
        check_docker
        require_running
        info "Launching single-instance PPO training..."
        info "Note: The simulation (Gazebo + Robot) must already be running in a separate terminal with './kairos_rl.sh sim'."
        if [ "$#" -eq 0 ]; then
            docker_exec "ros2 run kairos_rl train_ppo --total-timesteps 500000 --save-dir ./checkpoints/ --log-dir ./tb_logs/"
        else
            docker_exec "ros2 run kairos_rl train_ppo $*"
        fi
        ;;

    # ── TRAIN PARALLEL PPO ───────────────────────────────────────────────────
    train_parallel)
        check_docker
        require_running
        info "Launching parallel multi-environment PPO training..."
        if [ "$#" -eq 0 ]; then
            docker_exec "ros2 run kairos_rl train_ppo_parallel --num-envs 2 --total-timesteps 500000"
        else
            docker_exec "ros2 run kairos_rl train_ppo_parallel $*"
        fi
        ;;

    # ── EVALUATION / POLICY TEST ─────────────────────────────────────────────
    eval|eval_policy|test_policy)
        check_docker
        require_running
        info "Evaluating RL policy with 3D Gazebo visualization..."
        if [ "$#" -eq 0 ]; then
            info "Default test: Level 0 (Reach) on pretrained checkpoint 'level0_final.zip' (5 episodes)."
            docker_exec "ros2 run kairos_rl eval_curriculum --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 5"
        else
            docker_exec "ros2 run kairos_rl eval_curriculum $*"
        fi
        ;;

    # ── TENSORBOARD ──────────────────────────────────────────────────────────
    tb|tensorboard)
        check_docker
        require_running
        info "Launching TensorBoard server..."
        echo -e "${GREEN}${BOLD}👉 Open your browser at: http://localhost:6006${NC}\n"
        docker_exec "tensorboard --logdir ${ROS2_WS}/tb_logs --port 6006 --bind_all"
        ;;

    # ── MODULAR WORKFLOW — TERMINAL 1: GAZEBO WORLD ──────────────────────────
    world)
        check_docker
        require_running
        WORLD="${1:-labo}"
        GUI="${2:-true}"
        info "Launching Gazebo world '${WORLD}' alone in Terminal 1 (GUI: ${GUI})..."
        if [ -n "${DISPLAY:-}" ] && command -v xhost &>/dev/null; then
            xhost +local:docker >/dev/null 2>&1 || true
        fi
        docker_exec "ros2 launch kairos_bringup kairos_world.launch.py world:=${WORLD} gui:=${GUI}"
        ;;

    # ── MODULAR WORKFLOW — TERMINAL 2: ROBOT SPAWN & RVIZ ─────────────────────
    robot)
        check_docker
        require_running
        GRIPPER="${1:-schunk_egk50}"
        RAW_RVIZ="${2:-true}"
        LOW_PERF="${3:-false}"
        case "$(echo "${RAW_RVIZ}" | tr '[:upper:]' '[:lower:]')" in
            false|0|no|norviz|no-rviz|no_rviz|none) RUN_RVIZ="false" ;;
            *) RUN_RVIZ="true" ;;
        esac
        if [[ "${GRIPPER}" != "schunk_egk50" && "${GRIPPER}" != "tesollo_dg5f" ]]; then
            warn "Unsupported gripper '${GRIPPER}'. Only 'schunk_egk50' and 'tesollo_dg5f' are supported. Defaulting to 'schunk_egk50'."
            GRIPPER="schunk_egk50"
        fi
        info "Spawning RB-KAIROS robot (Gripper: ${GRIPPER}, RViz: ${RUN_RVIZ}, Low-Perf: ${LOW_PERF}) in Terminal 2..."
        if [ -n "${DISPLAY:-}" ] && command -v xhost &>/dev/null; then
            xhost +local:docker >/dev/null 2>&1 || true
        fi
        docker_exec "ros2 launch kairos_bringup kairos_robot.launch.py gripper_type:=${GRIPPER} run_rviz:=${RUN_RVIZ} low_performance_simulation:=${LOW_PERF}"
        ;;

    # ── STANDALONE RVIZ2 ─────────────────────────────────────────────────────
    rviz)
        check_docker
        require_running
        RVIZ_CFG="${1:-}"
        info "Launching standalone RViz2..."
        if [ -n "${DISPLAY:-}" ] && command -v xhost &>/dev/null; then
            xhost +local:docker >/dev/null 2>&1 || true
        fi
        if [ -n "${RVIZ_CFG}" ]; then
            docker_exec "ros2 launch kairos_bringup kairos_rviz.launch.py rviz_config:=${RVIZ_CFG}"
        else
            docker_exec "ros2 launch kairos_bringup kairos_rviz.launch.py"
        fi
        ;;

    # ── STANDALONE SIMULATION (VISUAL DEBUG SHORTCUT) ────────────────────────
    sim)
        check_docker
        require_running
        WORLD="${1:-labo}"
        GRIPPER="${2:-schunk_egk50}"
        if [[ "${GRIPPER}" != "schunk_egk50" && "${GRIPPER}" != "tesollo_dg5f" ]]; then
            warn "Unsupported gripper '${GRIPPER}'. Only 'schunk_egk50' and 'tesollo_dg5f' are supported. Defaulting to 'schunk_egk50'."
            GRIPPER="schunk_egk50"
        fi
        info "Launching standalone Gazebo simulation (World: ${WORLD}, Gripper: ${GRIPPER})..."
        info "Note: For Curriculum Learning, 'train' launches Gazebo automatically."
        docker_exec "ros2 launch kairos_bringup kairos_sim_complete.launch.py world:=${WORLD} gripper_type:=${GRIPPER}"
        ;;


    # ── TELEOP ───────────────────────────────────────────────────────────────
    teleop)
        check_docker
        require_running
        MODE="${1:-direct}"
        info "Launching RB-KAIROS keyboard teleoperation (arrows & speeds)..."
        info "Controls: Arrows (or Z/Q/S/D), Shift+Arrows (crabbing), +/- (speed), Space (emergency stop)"
        if [ "$MODE" = "xterm" ]; then
            info "Opening dedicated xterm window..."
            docker_exec "ros2 launch kairos_bringup kairos_teleop.launch.py use_xterm:=true"
        else
            info "Active control directly in this terminal (for xterm window: ./kairos_rl.sh teleop xterm)..."
            docker_exec "ros2 launch kairos_bringup kairos_teleop.launch.py use_xterm:=false"
        fi
        ;;

    # ── UNIT TESTS ───────────────────────────────────────────────────────────
    test)
        check_docker
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            info "Running unit tests inside container..."
            docker_exec "python3 -m pytest src/kairos_rl/test/test_kinematics_and_env.py src/kairos_bringup/test/ -o cache_dir=/tmp/.pytest_cache -v"
        elif command -v pytest &>/dev/null || python3 -m pytest --version &>/dev/null; then
            info "Container not running: running unit tests on host machine..."
            python3 -m pytest "${SCRIPT_DIR}/src/kairos_rl/test/test_kinematics_and_env.py" "${SCRIPT_DIR}/src/kairos_bringup/test/" -o cache_dir=/tmp/.pytest_cache -v
        else
            require_running
            docker_exec "python3 -m pytest src/kairos_rl/test/test_kinematics_and_env.py src/kairos_bringup/test/ -o cache_dir=/tmp/.pytest_cache -v"
        fi
        ;;

    # ── CLEAN (CACHE, IPC & STALE PROCESSES) ──────────────────────────────────
    clean)
        check_docker
        info "Cleaning FastRTPS residual IPC, shared memory descriptors, and orphaned processes..."
        rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            docker exec -u root "${CONTAINER_NAME}" bash -c "rm -rf /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /home/robot/.ros/log/* /root/.ros/log/* /tmp/gz* /tmp/rl_cube* 2>/dev/null || true"
            docker exec -u root "${CONTAINER_NAME}" bash -c "pkill -9 -f rviz2 || true; pkill -9 -f gz || true; pkill -9 -f 'ros2 launch' || true; pkill -9 -f robot_state_publisher || true; pkill -9 -f parameter_bridge || true; pkill -9 -f spawner || true; pkill -9 -f train_ppo || true; pkill -9 -f train_curriculum || true" 2>/dev/null || true
        fi
        success "Shared memory and residual processes cleaned."
        ;;

    # ── STATUS ───────────────────────────────────────────────────────────────
    status)
        check_docker
        echo -e "\n${BOLD}=== Container Status ===${NC}"
        docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo -e "\n${BOLD}=== Available Checkpoints (checkpoints/) ===${NC}"
        ls -lh "${SCRIPT_DIR}/checkpoints" 2>/dev/null || echo "No checkpoints in checkpoints/"
        echo -e "\n${BOLD}=== RL Level 0 Archive (rl_level0/) ===${NC}"
        COUNT=$(ls "${SCRIPT_DIR}/rl_level0" 2>/dev/null | wc -l || echo 0)
        echo "Total saved checkpoints in rl_level0/: ${COUNT}"
        if command -v nvidia-smi &>/dev/null; then
            echo -e "\n${BOLD}=== NVIDIA GPU Status ===${NC}"
            nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
        fi
        echo ""
        ;;

    # ── LOGS ─────────────────────────────────────────────────────────────────
    logs)
        check_docker
        info "Displaying container logs (Ctrl+C to quit)..."
        docker logs -f "${CONTAINER_NAME}"
        ;;

    # ── HELP ─────────────────────────────────────────────────────────────────
    help|--help|-h)
        echo -e "
${CYAN}╔══════════════════════════════════════════════════════════════════════╗
║               Kairos WS RL — Complete Management Script              ║
║     Reinforcement Learning (PPO & Curriculum) for RB-KAIROS+         ║
╚══════════════════════════════════════════════════════════════════════╝${NC}

Usage: ${GREEN}./kairos_rl.sh <command> [arguments...]${NC}

${BOLD}Lifecycle Commands:${NC}
  ${YELLOW}build${NC}                Build Docker image (Jazzy + Gazebo Harmonic + PyTorch)
  ${YELLOW}start${NC}                Start container in background
  ${YELLOW}stop${NC}                 Stop container
  ${YELLOW}restart${NC}              Restart container
  ${YELLOW}shell${NC}                Open interactive bash shell inside container
  ${YELLOW}status${NC}               Display container, GPU, and checkpoint status
  ${YELLOW}logs${NC}                 Stream container logs live

${BOLD}Build & Test Commands:${NC}
  ${YELLOW}colcon_build [pkg]${NC}   Compile ROS 2 workspace (or a specific package)
  ${YELLOW}test${NC}                 Run unit tests (kinematics, collisions, config)
  ${YELLOW}clean${NC}                Clean FastRTPS SHM (/dev/shm), logs, and zombies

${BOLD}Modular Multi-Terminal Simulation Commands:${NC}
  ${YELLOW}world [world] [gui]${NC}       Terminal 1: Launch Gazebo world alone (default: labo)
  ${YELLOW}robot [gripper] [rviz]${NC}    Terminal 2: Spawn RB-KAIROS robot + RViz2
                            gripper: schunk_egk50 (default) | tesollo_dg5f
                            rviz: true (default) | false
  ${YELLOW}rviz [config]${NC}             Terminal 2/4: Launch standalone RViz2
  ${YELLOW}sim [world] [gripper]${NC}     Combined launch (Gazebo + Robot + RViz)
                            gripper: schunk_egk50 (default) | tesollo_dg5f
  ${YELLOW}teleop [direct|xterm]${NC}     Terminal 3: Keyboard teleoperation of mobile base

${BOLD}Reinforcement Learning (RL) Commands (Terminal 3):${NC}
  ${YELLOW}train [options]${NC}      Launch Curriculum Learning training (4 Levels):
                         Examples:
                         ./kairos_rl.sh train --level 0 --num-envs 4 --total-timesteps 500000
                         ./kairos_rl.sh train --level 1 --resume ./checkpoints/level0_final.zip --num-envs 8
                         ./kairos_rl.sh train --level 2 --resume ./checkpoints/kairos_level1_pick_1000000_steps.zip
                         ./kairos_rl.sh train --level 3 --resume ./checkpoints/level2_final.zip

  ${YELLOW}train_ppo [options]${NC}  Launch standard single-instance PPO training
  ${YELLOW}train_parallel [opts]${NC} Launch parallel multi-process PPO training
  ${YELLOW}eval [options]${NC}       Evaluate pretrained policy in Gazebo with 3D visualization:
                         Example:
                         ./kairos_rl.sh eval --level 0 --checkpoint ./checkpoints/level0_final.zip --episodes 5
  ${YELLOW}tensorboard${NC}          Launch TensorBoard on port 6006 (loss & reward curves)

${BOLD}Recommended Modular Workflow:${NC}
  Terminal 0 (Host once): ${GREEN}xhost +local:docker && ./kairos_rl.sh start${NC}
  Terminal 1:             ${GREEN}./kairos_rl.sh world labo${NC}
  Terminal 2:             ${GREEN}./kairos_rl.sh robot schunk_egk50${NC}  (or tesollo_dg5f)
  Terminal 3 (RL / Teleop):
    - Train Level 0:      ${GREEN}./kairos_rl.sh train --level 0 --num-envs 2${NC}
    - Evaluate policy:    ${GREEN}./kairos_rl.sh eval --level 0${NC}
    - Keyboard teleop:    ${GREEN}./kairos_rl.sh teleop${NC}

See ${MAGENTA}README.md${NC} and ${MAGENTA}BEGINNER_GUIDE.md${NC} for detailed instructions.
"
        ;;

    # ── UNKNOWN COMMAND ──────────────────────────────────────────────────────
    *)
        error "Unknown command: '${CMD}'. Use './kairos_rl.sh help' to see options."
        ;;

esac
