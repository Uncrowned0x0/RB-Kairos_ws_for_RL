#!/usr/bin/env python3
# Copyright 2026 Kamil BENMADI <kamil.benmadi@sigma-clermont.fr>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Safe interactive keyboard teleoperation node for RB-KAIROS mobile base.

Supports:
- Arrow keys + AZERTY / QWERTY / IJKL letters
- Mecanum lateral strafe (crabbing) via Shift+Arrows or U/O
- Interactive speed adjustments (+ / -)
- Spacebar / X / C / ESC emergency stop
- Deadman safety: automatic halt when motion keys are released (default 0.60s to accommodate typematic repeat delay)
- Multi-topic output:
  - geometry_msgs/Twist on cmd_vel
  - geometry_msgs/TwistStamped on robotnik_base_control/reference (with BEST_EFFORT QoS and base frame_id)
  - geometry_msgs/Twist on robotnik_base_control/reference_unstamped (for un-stamped controller setups)
- Direct terminal execution or ros2 launch execution with seamless /dev/tty fallback

Author: Kamil BENMADI <kamil.benmadi@sigma-clermont.fr>
"""

import os
import select
import sys
import termios
import threading
import time
import tty

from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


BANNER_TEMPLATE = """\
\033[1;36m╔══════════════════════════════════════════════════════════════╗
║        🎮 RB-KAIROS SAFE KEYBOARD TELEOP (AZERTY/QWERTY)     ║
╚══════════════════════════════════════════════════════════════╝\033[0m
 \033[1;32mMotion Commands (Deadman safety: stops on key release)\033[0m
   • \033[1mUp Arrow\033[0m    or \033[1mZ / W / I\033[0m : Forward
   • \033[1mDown Arrow\033[0m  or \033[1mS / K\033[0m     : Backward
   • \033[1mLeft Arrow\033[0m  or \033[1mQ / A / J\033[0m : Turn Left
   • \033[1mRight Arrow\033[0m or \033[1mD / L\033[0m     : Turn Right
   • \033[1mShift + Arrows\033[0m or \033[1mU / O\033[0m : Mecanum Lateral Strafe (Left / Right)

 \033[1;33mSpeed Adjustment (NEVER MOVES THE ROBOT)\033[0m
   • \033[1m+\033[0m or \033[1mP\033[0m : Increase speed (+0.02 m/s) [Ceiling = {max_lin_cm:.0f} cm/s]
   • \033[1m-\033[0m or \033[1mM\033[0m : Decrease speed (-0.02 m/s) [Floor = {min_lin_cm:.0f} cm/s]

 \033[1;31mImmediate Emergency Stop\033[0m
   • \033[1mSPACEBAR\033[0m or \033[1mX\033[0m : IMMEDIATE STOP (0 m/s)
   • \033[1mCtrl+C\033[0m or \033[1mESC\033[0m   : Clean exit safely halting the robot
"""


class KairosSafeKeyboardTeleop(Node):
    """ROS 2 Node for safe interactive keyboard teleoperation."""

    def __init__(self, out_stream=None):
        super().__init__('kairos_teleop_keyboard')
        self.out_stream = out_stream if out_stream is not None else sys.stdout

        # Parameter declarations
        self.declare_parameter('default_linear_speed', 0.50)
        self.declare_parameter('default_angular_speed', 1.00)
        self.declare_parameter('max_linear_speed', 2.00)
        self.declare_parameter('min_linear_speed', 0.05)
        self.declare_parameter('max_angular_speed', 3.00)
        self.declare_parameter('min_angular_speed', 0.10)
        self.declare_parameter('deadman_timeout', 0.60)
        self.declare_parameter('frame_id', 'robot_base_footprint')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('stamped_cmd_vel_topic', 'robotnik_base_control/reference')
        self.declare_parameter('unstamped_reference_topic', 'robotnik_base_control/reference_unstamped')
        self.declare_parameter('publish_stamped', True)
        self.declare_parameter('publish_unstamped', True)
        self.declare_parameter('publish_reference_unstamped', True)

        self.linear_speed = float(self.get_parameter('default_linear_speed').value)
        self.angular_speed = float(self.get_parameter('default_angular_speed').value)
        self.max_lin = float(self.get_parameter('max_linear_speed').value)
        self.min_lin = float(self.get_parameter('min_linear_speed').value)
        self.max_ang = float(self.get_parameter('max_angular_speed').value)
        self.min_ang = float(self.get_parameter('min_angular_speed').value)
        self.deadman_timeout = float(self.get_parameter('deadman_timeout').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.stamped_topic = str(self.get_parameter('stamped_cmd_vel_topic').value)
        self.unstamped_ref_topic = str(self.get_parameter('unstamped_reference_topic').value)
        self.publish_stamped = bool(self.get_parameter('publish_stamped').value)
        self.publish_unstamped = bool(self.get_parameter('publish_unstamped').value)
        self.publish_reference_unstamped = bool(self.get_parameter('publish_reference_unstamped').value)

        # QoS for TwistStamped matching mecanum_drive_controller
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publisher for standard Twist (used by real robot / Nav2 cmd_vel / RViz)
        if self.publish_unstamped and self.cmd_vel_topic:
            self.pub_unstamped = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        else:
            self.pub_unstamped = None

        # Publisher for TwistStamped (used by simulation mecanum_drive_controller with use_stamped_vel: true)
        if self.publish_stamped and self.stamped_topic:
            self.pub_stamped = self.create_publisher(TwistStamped, self.stamped_topic, qos_best_effort)
        else:
            self.pub_stamped = None

        # Publisher for Twist on reference_unstamped (used by simulation mecanum_drive_controller with use_stamped_vel: false)
        if self.publish_reference_unstamped and self.unstamped_ref_topic:
            self.pub_ref_unstamped = self.create_publisher(Twist, self.unstamped_ref_topic, 10)
        else:
            self.pub_ref_unstamped = None

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_key_time = 0.0
        self.action_desc = 'STOPPED (Ready)'
        self.running = True

        # 20Hz cyclic publishing loop
        self.timer = self.create_timer(0.05, self.publish_twist)

    def publish_twist(self):
        """Publish twist to topics with deadman timeout enforcement."""
        now = time.time()
        # Enforce deadman timeout: if no key was pressed recently, reset velocities to 0
        if (now - self.last_key_time) > self.deadman_timeout:
            if self.vx != 0.0 or self.vy != 0.0 or self.wz != 0.0:
                self.vx = 0.0
                self.vy = 0.0
                self.wz = 0.0
                self.action_desc = 'STOPPED (Key released)'
                self.print_status()

        twist = Twist()
        twist.linear.x = float(self.vx)
        twist.linear.y = float(self.vy)
        twist.angular.z = float(self.wz)

        if self.pub_unstamped is not None:
            self.pub_unstamped.publish(twist)

        if self.pub_ref_unstamped is not None:
            self.pub_ref_unstamped.publish(twist)

        if self.pub_stamped is not None:
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = self.frame_id
            stamped.twist = twist
            self.pub_stamped.publish(stamped)

    def stop_robot(self, reason='EMERGENCY STOP'):
        """Immediately reset all target velocities to zero."""
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_key_time = 0.0
        self.action_desc = reason
        self.publish_twist()
        self.print_status()

    def adjust_speed(self, delta_lin, delta_ang):
        """Modify speed settings safely without sending motion."""
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.linear_speed = max(self.min_lin, min(self.max_lin, self.linear_speed + delta_lin))
        self.angular_speed = max(self.min_ang, min(self.max_ang, self.angular_speed + delta_ang))
        self.action_desc = f'Speed set to {self.linear_speed * 100:.1f} cm/s'
        self.publish_twist()
        self.print_status()

    def handle_command(self, vx_factor, vy_factor, wz_factor, desc):
        """Process directional movement command."""
        self.last_key_time = time.time()
        self.vx = vx_factor * self.linear_speed
        self.vy = vy_factor * self.linear_speed
        self.wz = wz_factor * self.angular_speed
        self.action_desc = desc
        self.print_status()

    def print_status(self):
        """Display single-line real-time dashboard in terminal."""
        bar_len = 10
        span = max(0.001, (self.max_lin - self.min_lin))
        ratio = max(0.0, min(1.0, (self.linear_speed - self.min_lin) / span))
        fill = int(round(ratio * bar_len))
        gauge = '█' * fill + '░' * (bar_len - fill)

        status_line = (
            f'[\033[1;33m{gauge}\033[0m '
            f'\033[1mSpeed: {self.linear_speed * 100:.1f} cm/s\033[0m | '
            f'Rot: {self.angular_speed:.2f} rad/s] -> '
            f'\033[1;32m{self.action_desc}\033[0m\n'
        )
        try:
            self.out_stream.write(status_line)
            self.out_stream.flush()
        except Exception:
            pass


def read_key(fd):
    """Read a key sequence from standard input non-blockingly."""
    r, _, _ = select.select([fd], [], [], 0.05)
    if not r:
        return None

    try:
        ch = os.read(fd, 1).decode('latin-1', errors='ignore')
    except Exception:
        return None

    if ch == '\x1b':
        # Potential escape sequence
        r2, _, _ = select.select([fd], [], [], 0.05)
        if r2:
            try:
                ch2 = os.read(fd, 1).decode('latin-1', errors='ignore')
            except Exception:
                return '\x1b'
            if ch2 == '[':
                seq = ''
                while True:
                    r3, _, _ = select.select([fd], [], [], 0.02)
                    if not r3:
                        break
                    try:
                        c = os.read(fd, 1).decode('latin-1', errors='ignore')
                    except Exception:
                        break
                    seq += c
                    if c.isalpha() or c == '~':
                        break
                return '\x1b[' + seq
            elif ch2 == 'O':  # SS3 sequence (some terminals send \x1bOA, \x1bOB for arrows)
                r3, _, _ = select.select([fd], [], [], 0.02)
                if r3:
                    try:
                        c = os.read(fd, 1).decode('latin-1', errors='ignore')
                        return '\x1bO' + c
                    except Exception:
                        return '\x1bO'
                return '\x1bO'
            return '\x1b' + ch2
    return ch


def main():
    """Run the keyboard teleoperation node."""
    tty_file = None
    tty_out = None

    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
    else:
        # Fallback to controlling terminal /dev/tty if stdin is piped (e.g. under ros2 launch)
        try:
            tty_file = open('/dev/tty', 'r')
            fd = tty_file.fileno()
        except Exception:
            print('Error: kairos_teleop_keyboard must be run in an interactive terminal (or have /dev/tty).')
            return

    # Determine interactive output stream (use /dev/tty if stdout is piped by launch system)
    try:
        if sys.stdout.isatty():
            out_stream = sys.stdout
        else:
            tty_out = open('/dev/tty', 'w')
            out_stream = tty_out
    except Exception:
        out_stream = sys.stdout

    old_settings = termios.tcgetattr(fd)

    rclpy.init()
    node = KairosSafeKeyboardTeleop(out_stream=out_stream)

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    # Clear terminal cleanly if supported
    term_env = os.environ.get('TERM', '')
    if term_env and term_env != 'dumb':
        try:
            os.system('clear')
        except Exception:
            out_stream.write('\033[2J\033[H')
    else:
        out_stream.write('\033[2J\033[H')

    banner = BANNER_TEMPLATE.format(max_lin_cm=node.max_lin * 100, min_lin_cm=node.min_lin * 100)
    out_stream.write(banner + '\n')
    out_stream.flush()
    node.print_status()

    try:
        tty.setraw(fd)
        while rclpy.ok() and node.running:
            key = read_key(fd)
            if key is None:
                continue

            # Ctrl+C or Ctrl+D
            if key in ('\x03', '\x04'):
                break

            # Emergency Stop keys: Spacebar or X or C or ESC
            elif key in (' ', 'x', 'X', 'c', 'C', '\x1b'):
                node.stop_robot('EMERGENCY STOP (SPACEBAR / ESC)')

            # Speed Adjustments (+ / - / p / m)
            elif key in ('+', 'p', 'P', '='):
                node.adjust_speed(+0.05, +0.10)
            elif key in ('-', 'm', 'M'):
                node.adjust_speed(-0.05, -0.10)

            # Directional - Arrow Keys (CSI and SS3)
            elif key in ('\x1b[A', '\x1bOA'):  # Up Arrow
                node.handle_command(1.0, 0.0, 0.0, '▲ FORWARD')
            elif key in ('\x1b[B', '\x1bOB'):  # Down Arrow
                node.handle_command(-1.0, 0.0, 0.0, '▼ BACKWARD')
            elif key in ('\x1b[D', '\x1bOD'):  # Left Arrow
                node.handle_command(0.0, 0.0, 1.0, '◄ TURN LEFT')
            elif key in ('\x1b[C', '\x1bOC'):  # Right Arrow
                node.handle_command(0.0, 0.0, -1.0, '► TURN RIGHT')

            # Shift + Arrow Keys (Crabbing Mecanum)
            elif key in ('\x1b[1;2D', '\x1b[2D', '\x1b[D;2', '\x1b[d'):  # Shift+Left / Crab Left
                node.handle_command(0.0, 1.0, 0.0, '◄◄ STRAFE LEFT')
            elif key in ('\x1b[1;2C', '\x1b[2C', '\x1b[C;2', '\x1b[c'):  # Shift+Right / Crab Right
                node.handle_command(0.0, -1.0, 0.0, '►► STRAFE RIGHT')
            elif key in ('\x1b[1;2A', '\x1b[2A', '\x1b[A;2', '\x1b[a'):  # Shift+Up / Forward
                node.handle_command(1.0, 0.0, 0.0, '▲ FORWARD')
            elif key in ('\x1b[1;2B', '\x1b[2B', '\x1b[B;2', '\x1b[b'):  # Shift+Down / Backward
                node.handle_command(-1.0, 0.0, 0.0, '▼ BACKWARD')

            # Directional - Letters (AZERTY & QWERTY & IJKL)
            elif key in ('z', 'Z', 'w', 'W', 'i', 'I'):  # Forward
                node.handle_command(1.0, 0.0, 0.0, '▲ FORWARD')
            elif key in ('s', 'S', 'k', 'K'):  # Backward
                node.handle_command(-1.0, 0.0, 0.0, '▼ BACKWARD')
            elif key in ('q', 'Q', 'a', 'A', 'j'):  # Turn Left
                node.handle_command(0.0, 0.0, 1.0, '◄ TURN LEFT')
            elif key in ('d', 'D', 'l'):  # Turn Right
                node.handle_command(0.0, 0.0, -1.0, '► TURN RIGHT')

            # Crabbing Mecanum (U / O / Shift+J / Shift+L)
            elif key in ('u', 'U', 'J'):  # Lateral Left
                node.handle_command(0.0, 1.0, 0.0, '◄◄ STRAFE LEFT')
            elif key in ('o', 'O', 'L'):  # Lateral Right
                node.handle_command(0.0, -1.0, 0.0, '►► STRAFE RIGHT')

    except Exception as e:
        out_stream.write(f'\nTeleoperation error: {e}\n')
    finally:
        # Restore terminal settings
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        if tty_file is not None:
            try:
                tty_file.close()
            except Exception:
                pass

        # Transmit zero velocity multiple times for safety
        stop_twist = Twist()
        stop_stamped = TwistStamped()
        stop_stamped.header.stamp = node.get_clock().now().to_msg()
        stop_stamped.header.frame_id = node.frame_id
        for _ in range(5):
            if node.pub_unstamped is not None:
                node.pub_unstamped.publish(stop_twist)
            if node.pub_ref_unstamped is not None:
                node.pub_ref_unstamped.publish(stop_twist)
            if node.pub_stamped is not None:
                stop_stamped.header.stamp = node.get_clock().now().to_msg()
                node.pub_stamped.publish(stop_stamped)
            time.sleep(0.02)

        node.destroy_node()
        rclpy.shutdown()
        out_stream.write('\n\033[1;32mTeleoperation safely closed. Robot halted.\033[0m\n')
        out_stream.flush()
        if tty_out is not None:
            try:
                tty_out.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
