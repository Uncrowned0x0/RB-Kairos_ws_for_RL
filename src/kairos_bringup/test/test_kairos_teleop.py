#!/usr/bin/env python3
"""
Unit tests for RB-KAIROS safe keyboard teleoperation node.
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock

# Ensure ROS_LOG_DIR is writable in sandboxed/headless test environments
if 'ROS_LOG_DIR' not in os.environ:
    os.environ['ROS_LOG_DIR'] = '/tmp'

# Dynamically add script directory to sys.path so tests can be run from anywhere
_script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Also support importing when installed or relative to package root
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import kairos_teleop_keyboard as teleop
from geometry_msgs.msg import Twist, TwistStamped
import rclpy


class TestKairosTeleopKeyboard(unittest.TestCase):
    """Test suite for KairosSafeKeyboardTeleop node and key handling."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = teleop.KairosSafeKeyboardTeleop()

    def tearDown(self):
        self.node.destroy_node()

    def test_node_parameters_and_defaults(self):
        """Verify parameter declarations and default speed/timeout values."""
        self.assertEqual(self.node.linear_speed, 0.08)
        self.assertEqual(self.node.angular_speed, 0.20)
        self.assertEqual(self.node.max_lin, 0.15)
        self.assertEqual(self.node.min_lin, 0.03)
        self.assertEqual(self.node.max_ang, 0.30)
        self.assertEqual(self.node.min_ang, 0.05)
        self.assertEqual(self.node.deadman_timeout, 0.60)
        self.assertEqual(self.node.frame_id, 'robot_base_footprint')
        self.assertEqual(self.node.cmd_vel_topic, 'cmd_vel')
        self.assertEqual(self.node.stamped_topic, 'robotnik_base_control/reference')
        self.assertEqual(self.node.unstamped_ref_topic, 'robotnik_base_control/reference_unstamped')
        self.assertTrue(self.node.publish_stamped)
        self.assertTrue(self.node.publish_unstamped)
        self.assertTrue(self.node.publish_reference_unstamped)

    def test_publishers_created(self):
        """Verify that Twist, TwistStamped and reference_unstamped publishers are active."""
        self.assertIsNotNone(self.node.pub_unstamped)
        self.assertEqual(self.node.pub_unstamped.msg_type, Twist)
        self.assertIsNotNone(self.node.pub_ref_unstamped)
        self.assertEqual(self.node.pub_ref_unstamped.msg_type, Twist)
        self.assertIsNotNone(self.node.pub_stamped)
        self.assertEqual(self.node.pub_stamped.msg_type, TwistStamped)

    def test_handle_command_motion(self):
        """Test directional motion calculations for linear, strafe and angular."""
        # Forward
        self.node.handle_command(1.0, 0.0, 0.0, '▲ FORWARD')
        self.assertAlmostEqual(self.node.vx, 0.08)
        self.assertAlmostEqual(self.node.vy, 0.0)
        self.assertAlmostEqual(self.node.wz, 0.0)
        self.assertEqual(self.node.action_desc, '▲ FORWARD')

        # Backward
        self.node.handle_command(-1.0, 0.0, 0.0, '▼ BACKWARD')
        self.assertAlmostEqual(self.node.vx, -0.08)
        self.assertAlmostEqual(self.node.vy, 0.0)

        # Strafe Left (Mecanum)
        self.node.handle_command(0.0, 1.0, 0.0, '◄◄ STRAFE LEFT')
        self.assertAlmostEqual(self.node.vx, 0.0)
        self.assertAlmostEqual(self.node.vy, 0.08)
        self.assertAlmostEqual(self.node.wz, 0.0)

        # Turn Right
        self.node.handle_command(0.0, 0.0, -1.0, '► TURN RIGHT')
        self.assertAlmostEqual(self.node.vx, 0.0)
        self.assertAlmostEqual(self.node.vy, 0.0)
        self.assertAlmostEqual(self.node.wz, -0.20)

    def test_adjust_speed_limits(self):
        """Test speed adjustment increments, ceilings, and floors."""
        # Increase speed
        initial_speed = self.node.linear_speed
        self.node.adjust_speed(+0.02, +0.03)
        self.assertAlmostEqual(self.node.linear_speed, initial_speed + 0.02)
        # Motion must be zeroed when changing speed
        self.assertEqual(self.node.vx, 0.0)
        self.assertEqual(self.node.vy, 0.0)
        self.assertEqual(self.node.wz, 0.0)

        # Hit ceiling
        for _ in range(10):
            self.node.adjust_speed(+0.05, +0.05)
        self.assertAlmostEqual(self.node.linear_speed, self.node.max_lin)
        self.assertAlmostEqual(self.node.angular_speed, self.node.max_ang)

        # Hit floor
        for _ in range(20):
            self.node.adjust_speed(-0.05, -0.05)
        self.assertAlmostEqual(self.node.linear_speed, self.node.min_lin)
        self.assertAlmostEqual(self.node.angular_speed, self.node.min_ang)

    def test_stop_robot(self):
        """Test immediate stop functionality."""
        self.node.handle_command(1.0, 0.0, 0.0, 'FORWARD')
        self.assertNotEqual(self.node.vx, 0.0)
        self.node.stop_robot('TEST STOP')
        self.assertEqual(self.node.vx, 0.0)
        self.assertEqual(self.node.vy, 0.0)
        self.assertEqual(self.node.wz, 0.0)
        self.assertEqual(self.node.action_desc, 'TEST STOP')

    def test_deadman_timeout(self):
        """Test that deadman timeout resets target velocities to zero."""
        self.node.handle_command(1.0, 0.0, 0.0, 'FORWARD')
        self.assertAlmostEqual(self.node.vx, 0.08)
        # Simulate elapsed time past deadman_timeout
        self.node.last_key_time = time.time() - 1.0
        self.node.publish_twist()
        self.assertEqual(self.node.vx, 0.0)
        self.assertEqual(self.node.vy, 0.0)
        self.assertEqual(self.node.wz, 0.0)
        self.assertIn('STOPPED', self.node.action_desc)

    def test_multi_topic_publishing_content(self):
        """Verify published messages on unstamped, ref_unstamped, and stamped publishers."""
        # Mock publishers to inspect published message content
        self.node.pub_unstamped = MagicMock()
        self.node.pub_ref_unstamped = MagicMock()
        self.node.pub_stamped = MagicMock()

        self.node.vx = 0.08
        self.node.vy = 0.05
        self.node.wz = 0.15
        self.node.last_key_time = time.time()

        self.node.publish_twist()

        # Check unstamped Twist message on cmd_vel
        self.node.pub_unstamped.publish.assert_called_once()
        published_twist = self.node.pub_unstamped.publish.call_args[0][0]
        self.assertIsInstance(published_twist, Twist)
        self.assertAlmostEqual(published_twist.linear.x, 0.08)
        self.assertAlmostEqual(published_twist.linear.y, 0.05)
        self.assertAlmostEqual(published_twist.angular.z, 0.15)

        # Check unstamped Twist message on reference_unstamped
        self.node.pub_ref_unstamped.publish.assert_called_once()
        published_ref = self.node.pub_ref_unstamped.publish.call_args[0][0]
        self.assertIsInstance(published_ref, Twist)
        self.assertAlmostEqual(published_ref.linear.x, 0.08)
        self.assertAlmostEqual(published_ref.linear.y, 0.05)
        self.assertAlmostEqual(published_ref.angular.z, 0.15)

        # Check TwistStamped message on reference
        self.node.pub_stamped.publish.assert_called_once()
        published_stamped = self.node.pub_stamped.publish.call_args[0][0]
        self.assertIsInstance(published_stamped, TwistStamped)
        self.assertEqual(published_stamped.header.frame_id, 'robot_base_footprint')
        self.assertAlmostEqual(published_stamped.twist.linear.x, 0.08)
        self.assertAlmostEqual(published_stamped.twist.linear.y, 0.05)
        self.assertAlmostEqual(published_stamped.twist.angular.z, 0.15)

    def test_key_actions_mapping(self):
        """Verify that all arrow and letter keys trigger expected motion directions."""
        # Up Arrow
        self.node.handle_command(1.0, 0.0, 0.0, '▲ FORWARD')
        self.assertGreater(self.node.vx, 0.0)
        self.assertEqual(self.node.vy, 0.0)

        # Down Arrow
        self.node.handle_command(-1.0, 0.0, 0.0, '▼ BACKWARD')
        self.assertLess(self.node.vx, 0.0)
        self.assertEqual(self.node.vy, 0.0)

        # Left Arrow
        self.node.handle_command(0.0, 0.0, 1.0, '◄ TURN LEFT')
        self.assertEqual(self.node.vx, 0.0)
        self.assertGreater(self.node.wz, 0.0)

        # Right Arrow
        self.node.handle_command(0.0, 0.0, -1.0, '► TURN RIGHT')
        self.assertEqual(self.node.vx, 0.0)
        self.assertLess(self.node.wz, 0.0)

        # Shift+Left (Strafe Left)
        self.node.handle_command(0.0, 1.0, 0.0, '◄◄ STRAFE LEFT')
        self.assertEqual(self.node.vx, 0.0)
        self.assertGreater(self.node.vy, 0.0)

        # Shift+Right (Strafe Right)
        self.node.handle_command(0.0, -1.0, 0.0, '►► STRAFE RIGHT')
        self.assertEqual(self.node.vx, 0.0)
        self.assertLess(self.node.vy, 0.0)


if __name__ == '__main__':
    unittest.main()
