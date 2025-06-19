import sys
sys.path.append("./")

import numpy as np

from sensor.teleoperation_sensor import TeleoperationSensor
from utils.ros_subscriber import ROSSubscriber 
from utils.data_handler import compute_local_delta_pose, debug_print, compute_rotate_matrix

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from typing import Callable, Optional

'''
Pika base code(ROS) from:
https://github.com/agilexrobotics/pika_ros.git
'''

class PikaSensor(TeleoperationSensor):
    def __init__(self,name):
        super().__init__()
        self.name = name
    
    def set_up(self, pos_node_name, gripper_node_name, call: Optional[Callable] = None):

        self.pos_subscriber = ROSSubscriber(pos_node_name, PoseStamped, call)

        self.gripper_subscriber = ROSSubscriber(gripper_node_name, JointState)
        
        self.sensor = {
            "pos_subscriber":self.pos_subscriber,
            "gripper_subscriber":self.gripper_subscriber,
        }

        self.prev_qpos = None

    def get_state(self):
        pos_msg = self.sensor["pos_subscriber"].get_latest_data()
        if pos_msg is None:
            qpos = None
            debug_print(f"{self.name}", f"getting message pose from pika error!", "ERROR")
        else:
            roll, pitch, yaw = R.from_quat([pos_msg.pose.orientation.x,pos_msg.pose.orientation.y, \
                                            pos_msg.pose.orientation.z,pos_msg.pose.orientation.w]).as_euler('xyz')
            qpos = np.array([pos_msg.pose.position.x,
                    pos_msg.pose.position.y,
                    pos_msg.pose.position.z,
                    roll,
                    pitch,
                    yaw,])

        if self.prev_qpos is None:
            self.prev_qpos = qpos
            qpos = np.array([0,0,0,0,0,0])
        else:
            qpos = compute_local_delta_pose(self.prev_qpos, qpos)
        
        # gripper_msg = self.sensor["gripper_subscriber"].get_latest_data()
        # if gripper_msg is None:
        #     gripper = None
        #     debug_print(f"{self.name}", f"getting message gripper from pika error!", "ERROR")
        # else:
        # # 归一化
        #     gripper = (np.array([gripper_msg.position])[0] - 0.3) / 1.7

        qpos = compute_rotate_matrix(qpos)
        return {
            "end_pose":qpos,
            # "gripper":gripper
        }

    def reset(self):
        pos_msg = self.sensor["pos_subscriber"].get_latest_data()
        roll, pitch, yaw = R.from_quat([pos_msg.pose.orientation.x,pos_msg.pose.orientation.y, \
                                            pos_msg.pose.orientation.z,pos_msg.pose.orientation.w]).as_euler('xyz')
        qpos = np.array([pos_msg.pose.position.x,
                pos_msg.pose.position.y,
                pos_msg.pose.position.z,
                roll,
                pitch,
                yaw,])
        
        self.prev_qpos = qpos
        debug_print(f"{self.name}", "reset success!", "INFO")

if __name__ == "__main__":
    import time
    import rospy
    pika_left = PikaSensor("left_pika")
    pika_right = PikaSensor("right_pika")

    pika_left.set_up("/pika_pose_l","/gripper_l/joint_states")
    pika_right.set_up("/pika_pose_r","/gripper_r/joint_states")

    pika_left.set_collect_info(["end_pose","gripper"])
    pika_right.set_collect_info(["end_pose","gripper"])

    rospy.init_node('ros_subscriber_node', anonymous=True)
    left_transform_matrix =   np.array([[0, 0, -1, 0.3],
                                        [-1, 0, 0, 0.3],
                                        [0, 1, 0, 0.1],
                                        [ 0, 0, 0, 1]])
    
    right_transform_matrix =   np.array([[0, 0, -1, 0.3],
                                        [1, 0, 0,-0.3],
                                        [0, -1, 0, 0.1],
                                        [0, 0, 0, 1]])
    
    def apply_fixed_transform(T_current, T_offset):
        R_current = T_current[:3, :3]
        t_current = T_current[:3, 3]

        R_target = T_offset[:3, :3] @ R_current
        t_target = T_offset[:3, 3] + t_current

        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = t_target
        return T_target
    
    while True:
        left_pose = pika_left.get_state()["end_pose"]
        right_pose = pika_right.get_state()["end_pose"]

        left_wrist_mat = apply_fixed_transform(left_pose, left_transform_matrix)
        right_wrist_mat = apply_fixed_transform(right_pose, right_transform_matrix)

        print("left_pika:\n", left_wrist_mat)
        print("right_pika:\n", right_wrist_mat)
        time.sleep(0.1)