"""
Unit tests for kinematics, collision detection, and KAIROS RL configurations.
Executable both inside the ROS 2 container and directly on the host machine.
"""
import os
import sys
import math
import yaml
import pytest
import numpy as np
from unittest.mock import MagicMock

# Add package directory to sys.path
PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

# Automatic mocking of ROS 2 modules if not installed on host
for mod in [
    'rclpy', 'rclpy.node', 'rclpy.duration', 'rclpy.executors', 'rclpy.qos',
    'sensor_msgs', 'sensor_msgs.msg', 'geometry_msgs', 'geometry_msgs.msg',
    'trajectory_msgs', 'trajectory_msgs.msg', 'visualization_msgs', 'visualization_msgs.msg',
    'nav_msgs', 'nav_msgs.msg', 'builtin_interfaces', 'builtin_interfaces.msg'
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from kairos_rl.curriculum.level0_reach import (
    ur5e_fk,
    ur5e_self_collision_check,
    check_robot_collisions,
    ReachEnv,
    ARM_MOUNT_Z,
    HOME_POSITION,
)


def test_ur5e_forward_kinematics_dimensions():
    """Verify that forward kinematics returns a valid 3D Cartesian vector."""
    q_zero = np.zeros(6)
    tcp_zero = ur5e_fk(q_zero)
    assert tcp_zero.shape == (3,)
    assert not np.isnan(tcp_zero).any()

    # HOME pose
    tcp_home = ur5e_fk(HOME_POSITION)
    assert tcp_home.shape == (3,)
    assert not np.isnan(tcp_home).any()
    # Arm at HOME must be within realistic physical distance from base (< 1.2m)
    dist = np.linalg.norm(tcp_home)
    assert 0.1 < dist < 1.2


def test_collision_detection_chassis_and_table():
    """Verify collision detection against mobile base chassis and table."""
    # Safe configuration (arm pointing upwards)
    safe_q = np.array([0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0])
    chassis_hit, table_hit, arm_hit, any_col, _ = check_robot_collisions(safe_q)
    assert not chassis_hit, "High pose should not hit chassis"
    assert not table_hit, "High pose should not hit table"

    # Pose with TCP beneath floor/table level
    low_q = np.array([0.0, 0.5, 2.0, 0.0, 0.0, 0.0])
    _, table_hit_low, _, any_col_low, _ = check_robot_collisions(low_q)
    assert any_col_low, "Pose beneath table/ground must be detected as collision"


def test_config_files_validity():
    """Verify that YAML configuration files exist and are well-formed."""
    config_dir = os.path.join(PACKAGE_DIR, 'config')
    curriculum_path = os.path.join(config_dir, 'curriculum_params.yaml')
    rl_params_path = os.path.join(config_dir, 'rl_params.yaml')

    assert os.path.exists(curriculum_path), f"Missing file: {curriculum_path}"
    assert os.path.exists(rl_params_path), f"Missing file: {rl_params_path}"

    with open(curriculum_path, 'r') as f:
        curr_cfg = yaml.safe_load(f)
    assert 'curriculum' in curr_cfg
    assert 'environment' in curr_cfg
    assert 'reward' in curr_cfg
    assert 'ppo' in curr_cfg

    with open(rl_params_path, 'r') as f:
        rl_cfg = yaml.safe_load(f)
    assert 'environment' in rl_cfg
    assert 'arm' in rl_cfg
    assert 'reward' in rl_cfg


def test_reach_target_sampling_bounds():
    """Verify target sampling bounds generate valid targets in frontal workspace."""
    class DummyReach(ReachEnv):
        def __init__(self):
            self.spawn_radius_range = [0.30, 0.70]
            self.spawn_angle_range = [-1.2, 1.2]
            self.spawn_z = 0.76

    env = DummyReach()
    for _ in range(100):
        target = env._sample_target()
        assert target.shape == (3,)
        # X must be positive (frontal workspace) since cos([-1.2, 1.2]) > 0
        assert target[0] > 0.0, f"Target X must be positive: {target[0]}"
        # Z must be above table
        assert target[2] > 0.75, f"Target Z must be above table: {target[2]}"
        radius = math.sqrt(target[0]**2 + target[1]**2)
        assert 0.29 <= radius <= 0.71, f"Radius out of bounds: {radius}"


def test_dynamic_xacro_resolver():
    """Verify that XACRO resolver finds the ground-anchored RL model."""
    from kairos_rl.eval_curriculum import get_rl_xacro_path
    xacro_path = get_rl_xacro_path()
    assert os.path.exists(xacro_path), f"XACRO file not found: {xacro_path}"
    with open(xacro_path, 'r') as f:
        content = f.read()
    assert "world_to_base" in content, "RL model must include fixed joint 'world_to_base'"
    assert "schunk_egk50" in content, "RL model must support Schunk EGK50 gripper"


def test_checkpoint_loadable_and_architecture():
    """Verify integrity and architecture of pretrained checkpoint level0_final.zip."""
    from stable_baselines3 import PPO
    ws_root = os.path.abspath(os.path.join(PACKAGE_DIR, '..', '..'))
    ckpt_path = os.path.join(ws_root, 'checkpoints', 'level0_final.zip')
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = PPO.load(ckpt_path, env=None, device='cpu')
    assert model.observation_space.shape == (15,)
    assert model.action_space.shape == (6,)
    assert hasattr(model, 'policy')


def test_eval_curriculum_script_components():
    """Verify consistency of evaluation dictionary mappings and loaders."""
    from kairos_rl.eval_curriculum import LEVEL_NAMES, load_config
    assert LEVEL_NAMES[0] == 'Reach'
    assert LEVEL_NAMES[1] == 'Pick'
    assert LEVEL_NAMES[2] == 'Place'
    assert LEVEL_NAMES[3] == 'Full'

    cfg = load_config('config/curriculum_params.yaml')
    assert isinstance(cfg, dict)
    assert 'curriculum' in cfg


def test_bash_compose_files_cpu_no_pollution():
    """Verify FORCE_CPU=1 does not pollute command substitution $(get_compose_files)."""
    import subprocess
    script_path = os.path.abspath(os.path.join(PACKAGE_DIR, '..', '..', 'kairos_rl.sh'))
    cmd = (
        f"FORCE_CPU=1 bash -c 'source \"{script_path}\" help > /dev/null; "
        "COMPOSE_ARGS=($(get_compose_files)); echo \"${#COMPOSE_ARGS[@]}\"'"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert res.returncode == 0
    arg_count = int(res.stdout.strip())
    assert arg_count == 2, f"Expected 2 args (-f file.yaml), received {arg_count}"
