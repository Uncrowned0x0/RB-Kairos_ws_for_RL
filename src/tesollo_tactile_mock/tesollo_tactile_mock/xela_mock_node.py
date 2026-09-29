import rclpy
from rclpy.node import Node
import numpy as np

from ros_gz_interfaces.msg import Contacts
from geometry_msgs.msg import WrenchStamped

class XelaMockNode(Node):
    def __init__(self):
        super().__init__('xela_mock_node')
        
        self.num_tips = 4
        self.publishers_ = {}
        self.subscribers_ = []
        
        for i in range(1, self.num_tips + 1):
            # Subscribe to the contact topic bridged from Gazebo
            sub_topic = f'/contact/tip_{i}'
            sub = self.create_subscription(
                Contacts,
                sub_topic,
                lambda msg, finger_id=i: self.contact_callback(msg, finger_id),
                10
            )
            self.subscribers_.append(sub)
            
            # Publisher for the tactile wrench
            pub_topic = f'/tactile/finger_{i}'
            pub = self.create_publisher(WrenchStamped, pub_topic, 10)
            self.publishers_[i] = pub
            
        self.get_logger().info('Xela mock node initialized. Listening to /contact/tip_* and publishing to /tactile/finger_*')

    def contact_callback(self, msg, finger_id):
        wrench_msg = WrenchStamped()
        wrench_msg.header = msg.header
        # For Contacts from ros_gz_bridge, frame_id might be set appropriately
        
        # Aggregate forces from all contact points
        fx, fy, fz = 0.0, 0.0, 0.0
        
        for contact in msg.contact:
            # contact.wrench is an array of wrenches (one per body involved in the collision)
            # Typically [0] is the primary body
            for wrench in contact.wrench:
                fx += wrench.force.x
                fy += wrench.force.y
                fz += wrench.force.z
        
        # In a real scenario, you'd map these to the exact local frame of the sensor.
        # Assuming Gazebo returns the wrench in the link's frame (or we just pass it as is for RL):
        
        wrench_msg.wrench.force.x = fx
        wrench_msg.wrench.force.y = fy
        wrench_msg.wrench.force.z = fz
        
        # Torques might also be present if needed
        # wrench_msg.wrench.torque.x = ...
        
        self.publishers_[finger_id].publish(wrench_msg)

def main(args=None):
    rclpy.init(args=args)
    node = XelaMockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
