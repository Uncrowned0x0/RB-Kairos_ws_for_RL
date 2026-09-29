#!/usr/bin/env python3
"""
Relay node: converts Twist (from Nav2) to TwistStamped (for mecanum_drive_controller).

Nav2 in Jazzy always publishes geometry_msgs/Twist on cmd_vel.
The mecanum_drive_controller expects geometry_msgs/TwistStamped on ~/reference.
This node bridges the gap.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped


class TwistToStampedRelay(Node):
    def __init__(self):
        super().__init__('twist_to_stamped_relay')

        # QoS: match the motor controller's BEST_EFFORT subscription
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/robot/robotnik_base_control/reference',
            qos
        )

        self.subscription_ = self.create_subscription(
            Twist,
            '/robot/nav2_cmd_vel',
            self.twist_callback,
            10
        )

        self.get_logger().info(
            'Relay active: /robot/nav2_cmd_vel (Twist) → '
            '/robot/robotnik_base_control/reference (TwistStamped)'
        )

    def twist_callback(self, msg: Twist):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'robot_base_footprint'
        stamped.twist = msg
        self.publisher_.publish(stamped)


def main(args=None):
    rclpy.init(args=args)
    node = TwistToStampedRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
