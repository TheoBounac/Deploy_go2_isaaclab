import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped, Quaternion
from sensor_msgs.msg import PointCloud2, PointField, Image, JointState
import go2.go2_env as go2_env   # [go2_env.py]    
import go2.go2_ctrl as go2_ctrl # [go2_ctrl.py]   
import torch

class OdomListener(Node):
    def __init__(self):
        super().__init__('odom_listener')
        self.subscription = self.create_subscription(Odometry, '/unitree_go2/odom',
            self.listener_callback, 10)

    def listener_callback(self, msg):
        env_idx = 0

        lin = msg.twist.twist.linear
        if go2_env.base_lin_vel_input is not None:                            # [go2_env.py]                                       
            go2_env.base_lin_vel_input[env_idx][0] = msg.twist.twist.linear.x # [go2_env.py]                                       
            go2_env.base_lin_vel_input[env_idx][1] = msg.twist.twist.linear.y # [go2_env.py]                                       
            go2_env.base_lin_vel_input[env_idx][2] = msg.twist.twist.linear.z # [go2_env.py]                                       

        ang = msg.twist.twist.angular
        if go2_env.base_ang_vel_input is not None:
            go2_env.base_ang_vel_input[env_idx][0] = ang.x                    # [go2_env.py]                                       
            go2_env.base_ang_vel_input[env_idx][1] = ang.y                    # [go2_env.py]                                       
            go2_env.base_ang_vel_input[env_idx][2] = ang.z                    # [go2_env.py]                                       
  


class CmdListener(Node):
    def __init__(self):
        super().__init__('cmd_listener')
        self.subscription = self.create_subscription(Twist, 'unitree_go2/cmd_vel',
            self.listener_callback, 10)

    def listener_callback(self, msg):
        env_idx = 0                                       
        go2_ctrl.base_vel_cmd_input[env_idx][0] = msg.linear.x   # [go2_ctrl.py]                                       
        go2_ctrl.base_vel_cmd_input[env_idx][1] = msg.linear.y   # [go2_ctrl.py]                                       
        go2_ctrl.base_vel_cmd_input[env_idx][2] = msg.angular.z  # [go2_ctrl.py]                                       



class QuaternionListener(Node):
    def __init__(self):
        super().__init__('quaternion_listener')
        self.subscription = self.create_subscription(Quaternion, '/unitree_go2/quaternion',
            self.listener_callback, 10)

    def get_projected_gravity_from_quaternion(self,qw,qx,qy,qz):                            # [ROS2]
        gravity = torch.zeros(3)                                                            # [ROS2]
                                                                                            # [ROS2]
        # Applique rotation inverse de [0, 0, -1] dans le repère du robot                   # [ROS2]
        # Formule dérivée de R^T * [0, 0, -1] où R est la matrice de rotation du quaternion # [ROS2]
                                                                                            # [ROS2]
        gravity[0] = 2 * (-qx * qz + qw * qy)                                               # [ROS2]
        gravity[1] = -2 * (qy * qz + qw * qx)                                               # [ROS2]
        gravity[2] = -(1 - 2 * (qx**2 + qy**2))                                             # [ROS2]
                                                                                            # [ROS2]
        return gravity   

    def listener_callback(self, msg):
        env_idx = 0
        if go2_env.quaternion_input is not None:          # [go2_env.py]                                       
            go2_env.quaternion_input[env_idx][0] = msg.w  # [go2_env.py]                                       
            go2_env.quaternion_input[env_idx][1] = msg.x  # [go2_env.py]                                       
            go2_env.quaternion_input[env_idx][2] = msg.y  # [go2_env.py]                                       
            go2_env.quaternion_input[env_idx][3] = msg.z  # [go2_env.py]                                       
        
        if go2_env.gravity_input is not None:
            go2_env.gravity_input[env_idx] = self.get_projected_gravity_from_quaternion(msg.w,msg.x,msg.y,msg.z)
        


class JointListener(Node):
    def __init__(self):
        super().__init__('joint_listener')
        self.subscription = self.create_subscription(JointState, '/unitree_go2/lowstate', self.listener_callback, 10)

    def listener_callback(self, msg):
        env_idx = 0
        default_joint_pos = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]  
        if go2_env.joint_pos_input is not None:
            joint_pos = msg.position  # C’est une liste de float
            joint_pos = list(joint_pos)
            for i in range(12):
                go2_env.joint_pos_input[env_idx][i] = joint_pos[i]-default_joint_pos[i]
            
            joint_vel = msg.velocity
            for i in range(12):
                go2_env.joint_vel_input[env_idx][i] = joint_vel[i]




def main(args=None):
    rclpy.init(args=args)
    node = OdomListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


