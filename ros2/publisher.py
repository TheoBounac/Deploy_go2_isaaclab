#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3

class OdomZeroPublisher(Node):
    def __init__(self):
        super().__init__('odom_zero_publisher')
        self.publisher = self.create_publisher(Odometry, '/unitree_go2/odom', 10)
        self.timer = self.create_timer(5.0, self.timer_callback)  # ↩️ toutes les 5 secondes

    def timer_callback(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'unitree_go2/base_link'

        # Tout à zéro
        msg.pose.pose.position = Point(x=10.0, y=0.0, z=0.0)
        msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        msg.twist.twist.linear = Vector3(x=10.0, y=10.0, z=0.0)
        msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self.publisher.publish(msg)
        self.get_logger().info(f"Published ZERO odom.")

def main(args=None):
    rclpy.init(args=args)
    node = OdomZeroPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
