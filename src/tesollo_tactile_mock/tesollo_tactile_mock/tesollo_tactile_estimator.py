import rclpy
from rclpy.node import Node
import numpy as np
import message_filters

from geometry_msgs.msg import WrenchStamped, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

class TesolloTactileEstimator(Node):
    def __init__(self):
        super().__init__('tesollo_tactile_estimator')
        
        # Parameters
        self.declare_parameter('effort_threshold', 0.05)
        self.effort_threshold = self.get_parameter('effort_threshold').value
        
        # State
        self.d5_efforts = [0.0, 0.0, 0.0, 0.0]
        
        # Subscriptions
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        self.finger_subs = []
        for i in range(1, 5):
            sub = message_filters.Subscriber(self, WrenchStamped, f'tactile/finger_{i}')
            self.finger_subs.append(sub)
            
        self.ts = message_filters.ApproximateTimeSynchronizer(self.finger_subs, 10, 0.1)
        self.ts.registerCallback(self.wrench_callback)
        
        # Publishers
        self.tactile_pub = self.create_publisher(Float64MultiArray, 'tactile/all_fingers', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'tactile/visualization_markers', 10)
        
        self.get_logger().info("Tesollo Tactile Estimator initialized.")

    def joint_state_callback(self, msg):
        # Find efforts for D5 joints: rj_dg_5_1 to rj_dg_5_4
        for i, name in enumerate(msg.name):
            if name.startswith('rj_dg_5_'):
                try:
                    joint_idx = int(name.split('_')[-1]) - 1
                    if joint_idx >= 0 and joint_idx < 4:
                        if i < len(msg.effort):
                            self.d5_efforts[joint_idx] = msg.effort[i]
                except ValueError:
                    pass

    def wrench_callback(self, *wrench_msgs):
        all_forces = []
        thumb_f = np.array([0.0, 0.0, 0.0])
        sum_f24 = np.array([0.0, 0.0, 0.0])
        
        for i, msg in enumerate(wrench_msgs):
            f = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
            all_forces.append(f)
            if i == 0:
                thumb_f = f
            else:
                sum_f24 += f
                
        # Estimate Pinky (D5) Force
        # F_pinky_raw = - F_thumb - sum(F_i for i=2..4)
        pinky_raw = - thumb_f - sum_f24
        
        # Proprioceptive Gating
        max_d5_effort = max([abs(e) for e in self.d5_efforts])
        
        if max_d5_effort < self.effort_threshold:
            # D5 is moving freely, no contact
            f_pinky = np.array([0.0, 0.0, 0.0])
        else:
            # Rectify normal force (Z) to be compressive only (positive)
            f_z = max(0.0, pinky_raw[2])
            
            # Clip magnitude to 15 N
            xy_norm = np.linalg.norm(pinky_raw[0:2])
            f_pinky = np.array([pinky_raw[0], pinky_raw[1], f_z])
            norm = np.linalg.norm(f_pinky)
            if norm > 15.0:
                f_pinky = (f_pinky / norm) * 15.0
                
        all_forces.append(f_pinky)
        
        # Publish unified Float64MultiArray
        out_msg = Float64MultiArray()
        # Flat array of 15 elements
        flat_forces = []
        for f in all_forces:
            flat_forces.extend(f.tolist())
        out_msg.data = flat_forces
        self.tactile_pub.publish(out_msg)
        
        # Publish RViz Markers
        self.publish_markers(all_forces, wrench_msgs[0].header.stamp)

    def publish_markers(self, all_forces, stamp):
        marker_array = MarkerArray()
        
        for i, f in enumerate(all_forces):
            finger_idx = i + 1
            # In simulation, the link might be named prefix_rl_dg_X_tip. 
            # We assume 'rl_dg_{finger_idx}_tip' as the frame_id.
            # If a prefix is used, it might be necessary to adjust this.
            frame_id = f'rl_dg_{finger_idx}_tip'
            
            fx, fy, fz = f[0], f[1], f[2]
            
            # Normal Force Marker (Cylinder)
            normal_marker = Marker()
            normal_marker.header.frame_id = frame_id
            normal_marker.header.stamp = stamp
            normal_marker.ns = "normal_force"
            normal_marker.id = finger_idx
            normal_marker.type = Marker.CYLINDER
            normal_marker.action = Marker.ADD
            
            # Scale proportional to Fz
            radius = max(0.005, min(0.02, 0.005 + fz * 0.001))
            normal_marker.scale.x = radius
            normal_marker.scale.y = radius
            normal_marker.scale.z = 0.001  # Flat cylinder
            
            normal_marker.pose.position.z = 0.0
            
            # Color
            normal_marker.color.r = 1.0
            normal_marker.color.g = 0.0
            normal_marker.color.b = 0.0
            normal_marker.color.a = 0.8 if finger_idx < 5 else 0.4
            
            marker_array.markers.append(normal_marker)
            
            # Shear Force Marker (Arrow)
            shear_norm = np.linalg.norm([fx, fy])
            if shear_norm > 0.1:
                shear_marker = Marker()
                shear_marker.header.frame_id = frame_id
                shear_marker.header.stamp = stamp
                shear_marker.ns = "shear_force"
                shear_marker.id = finger_idx
                shear_marker.type = Marker.ARROW
                shear_marker.action = Marker.ADD
                
                # Start at center, point in shear direction
                p_start = Point(x=0.0, y=0.0, z=0.0)
                # Scale length
                scale_len = shear_norm * 0.005
                direction = np.array([fx, fy]) / shear_norm
                p_end = Point(x=direction[0]*scale_len, y=direction[1]*scale_len, z=0.0)
                
                shear_marker.points = [p_start, p_end]
                
                shear_marker.scale.x = 0.002 # shaft diameter
                shear_marker.scale.y = 0.004 # head diameter
                shear_marker.scale.z = 0.004 # head length
                
                shear_marker.color.r = 0.0
                shear_marker.color.g = 0.0
                shear_marker.color.b = 1.0
                shear_marker.color.a = 0.8 if finger_idx < 5 else 0.4
                
                marker_array.markers.append(shear_marker)
                
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TesolloTactileEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
