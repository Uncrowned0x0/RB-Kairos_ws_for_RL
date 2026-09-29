#!/usr/bin/env python3
"""
Unit tests for modular launch files, GPU compose configuration, and gripper filtering.
"""

import os
import unittest
import yaml

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BRINGUP_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
LAUNCH_DIR = os.path.join(BRINGUP_DIR, "launch")
WS_DIR = os.path.abspath(os.path.join(BRINGUP_DIR, "..", ".."))


class TestModularLaunchAndGrippers(unittest.TestCase):
    """Test suite for modular bringup launch files and configuration integrity."""

    def test_modular_launch_files_exist(self):
        """Verify all modular launch files exist in kairos_bringup/launch/."""
        expected_files = [
            "kairos_world.launch.py",
            "kairos_robot.launch.py",
            "kairos_rviz.launch.py",
            "kairos_nav2.launch.py",
            "kairos_teleop.launch.py",
            "kairos_sim_complete.launch.py",
        ]
        for f in expected_files:
            file_path = os.path.join(LAUNCH_DIR, f)
            self.assertTrue(os.path.isfile(file_path), f"Missing launch file: {f}")

    def test_launch_files_syntax_valid(self):
        """Verify that all Python launch files parse with zero syntax errors."""
        for f in os.listdir(LAUNCH_DIR):
            if f.endswith(".py"):
                path = os.path.join(LAUNCH_DIR, f)
                with open(path, "r", encoding="utf-8") as src:
                    code = src.read()
                try:
                    compile(code, path, "exec")
                except SyntaxError as e:
                    self.fail(f"Syntax error in {f}: {e}")

    def test_docker_compose_gpu_acceleration_configured(self):
        """Verify docker-compose.yaml contains NVIDIA GPU hardware acceleration reservations."""
        compose_path = os.path.join(WS_DIR, "docker-compose.yaml")
        self.assertTrue(os.path.isfile(compose_path), "docker-compose.yaml not found")
        with open(compose_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for GPU reservations and environment variables
        self.assertIn("runtime: nvidia", content)
        self.assertIn("driver: nvidia", content)
        self.assertIn("capabilities: [gpu]", content)
        self.assertIn("NVIDIA_VISIBLE_DEVICES: all", content)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES: all", content)
        self.assertIn("__GLX_VENDOR_LIBRARY_NAME: nvidia", content)

    def test_strict_gripper_support(self):
        """Verify only schunk_egk50 and tesollo_dg5f are supported, obsolete grippers excluded."""
        # 1. Check controller YAML does not contain obsolete controllers
        control_yaml = os.path.join(
            WS_DIR,
            "src",
            "robotnik",
            "robotnik_simulation",
            "robotnik_gazebo_ignition",
            "config",
            "profile",
            "rbkairos",
            "rbkairos_plus_ros2_control.yaml",
        )
        if os.path.isfile(control_yaml):
            with open(control_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Find controller_manager parameters under /** or at root
            cm_dict = data.get("/**", data)
            cm_params = cm_dict.get("controller_manager", {}).get("ros__parameters", {})
            self.assertNotIn("qbhand_synergy_controller", cm_params)
            self.assertNotIn("onrobot_rg6_controller", cm_params)
            self.assertIn("schunk_egk50_controller", cm_params)
            self.assertIn("tesollo_dg5f_controller", cm_params)

        # 2. Check kairos_sim_complete.launch.py default gripper
        sim_complete_path = os.path.join(LAUNCH_DIR, "kairos_sim_complete.launch.py")
        with open(sim_complete_path, "r", encoding="utf-8") as f:
            sim_content = f.read()
        self.assertIn('default_value="schunk_egk50"', sim_content)
        self.assertNotIn('default_value="qbhand"', sim_content)

    def test_robot_description_default_gripper(self):
        """Verify robot_description.launch.py defaults to schunk_egk50 and not qbhand."""
        desc_launch = os.path.join(
            WS_DIR,
            "src",
            "robotnik",
            "robotnik_description",
            "launch",
            "robot_description.launch.py",
        )
        self.assertTrue(os.path.isfile(desc_launch), f"File not found: {desc_launch}")
        with open(desc_launch, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("default_value='schunk_egk50'", content)
        self.assertNotIn("default_value='qbhand'", content)

    def test_xacro_clean_of_obsolete_grippers(self):
        """Verify RB-KAIROS URDF xacros do not contain unresolvable packages (qb_hand_description, onrobot_rg6_description)."""
        xacro_files = [
            os.path.join(
                WS_DIR,
                "src",
                "robotnik",
                "rbkairos_common",
                "rbkairos_description",
                "robots",
                "rbkairos_ur5_qbhand.urdf.xacro",
            ),
        ]
        rl_xacro = os.path.join(
            WS_DIR,
            "src",
            "robotnik",
            "rbkairos_common",
            "rbkairos_description",
            "robots",
            "rbkairos_ur5_rl.urdf.xacro",
        )
        if os.path.isfile(rl_xacro):
            xacro_files.append(rl_xacro)

        for xf in xacro_files:
            if os.path.isfile(xf):
                with open(xf, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertNotIn("$(find qb_hand_description)", content)
                self.assertNotIn("$(find onrobot_rg6_description)", content)

    def test_spawn_robot_launch_early_validation(self):
        """Verify spawn_robot.launch.py implements early gripper validation before robot_description inclusion."""
        spawn_path = os.path.join(
            WS_DIR,
            "src",
            "robotnik",
            "robotnik_simulation",
            "robotnik_gazebo_ignition",
            "launch",
            "spawn_robot.launch.py",
        )
        self.assertTrue(os.path.isfile(spawn_path))
        with open(spawn_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Validation must occur before robot_description IncludeLaunchDescription
        idx_validation = content.find("if gripper_type not in ['schunk_egk50', 'tesollo_dg5f']:")
        idx_robot_desc = content.find("FindPackageShare('robotnik_description'), '/launch/robot_description.launch.py'")
        self.assertGreater(idx_validation, 0, "Gripper validation check missing")
        self.assertGreater(idx_robot_desc, 0, "robot_description inclusion missing")
        self.assertLess(
            idx_validation,
            idx_robot_desc,
            "Gripper validation MUST happen before robot_description inclusion to prevent PackageNotFoundError"
        )

    def test_kairos_robot_launch_defaults(self):
        """Verify kairos_robot.launch.py defaults to schunk_egk50 and run_rviz=true."""
        robot_launch = os.path.join(LAUNCH_DIR, "kairos_robot.launch.py")
        with open(robot_launch, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('default_value="schunk_egk50"', content)
        self.assertIn('default_value="true"', content)

    def test_cli_scripts_gripper_filtering(self):
        """Verify kairos.sh / kairos_rl.sh contains strict gripper validation with fallback."""
        for script_name in ["kairos.sh", "kairos_rl.sh"]:
            script_path = os.path.join(WS_DIR, script_name)
            if os.path.isfile(script_path):
                with open(script_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertIn("schunk_egk50", content)
                self.assertIn("tesollo_dg5f", content)
                self.assertIn('GRIPPER="schunk_egk50"', content)

    def test_labo_world_sensors_plugin(self):
        """Verify labo.world has Sensors plugin with ogre2 render engine for GPU acceleration."""
        labo_path = os.path.join(
            WS_DIR,
            "src",
            "robotnik",
            "robotnik_simulation",
            "robotnik_gazebo_ignition",
            "worlds",
            "labo.world",
        )
        if os.path.isfile(labo_path):
            with open(labo_path, "r", encoding="utf-8") as f:
                labo_content = f.read()
            self.assertIn("gz::sim::systems::Sensors", labo_content)
            self.assertIn("<render_engine>ogre2</render_engine>", labo_content)


if __name__ == "__main__":
    unittest.main()
