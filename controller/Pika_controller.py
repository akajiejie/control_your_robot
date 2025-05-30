import sys
sys.path.append("./")

import numpy as np

from controller.teleoperation_controller import TeleoperationController
from utils.ros_subscriber import ROSSubscriber 
from utils.data_handler import compute_local_delta_pose, debug_print

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
# from sensor_tools import Gripper
from scipy.spatial.transform import Rotation as R
from typing import Callable, Optional

'''
Pika base code(ROS) from:
https://github.com/agilexrobotics/pika_ros.git
'''

class PikaController(TeleoperationController):
    def __init__(self,name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None
    
    def set_up(self, pos_node_name, gripper_node_name, call: Optional[Callable] = None):

        self.pos_subscriber = ROSSubscriber(pos_node_name, PoseStamped, call)

        self.gripper_subscriber = ROSSubscriber(gripper_node_name, JointState)
        
        self.controller = {
            "pos_subscriber":self.pos_subscriber,
            "gripper_subscriber":self.gripper_subscriber,
        }

    def get_state(self):
        pos_msg = self.controller["pos_subscriber"].get_latest_data()
        
        roll, pitch, yaw = R.from_quat([pos_msg.pose.orientation.x,pos_msg.pose.orientation.y, \
                                        pos_msg.pose.orientation.z,pos_msg.pose.orientation.w]).as_euler('xyz')
        qpos = np.array([pos_msg.pose.position.x,
                pos_msg.pose.position.y,
                pos_msg.pose.position.z,
                roll,
                pitch,
                yaw,])
        
        gripper_msg = self.controller["gripper_subscriber"].get_latest_data()
        # 归一化
        gripper = (np.array([gripper_msg.position])[0] - 0.3) / 1.7
        if qpos is None:
            qpos = -1
        if gripper is None:
            gripper = -1
        print("end_pose", qpos)
        print("gripper", gripper)
        
        return {
            "end_pose":qpos,
            "gripper":gripper
        }

if __name__ == "__main__":
    import time
    import rospy
    pika_left = PikaController("left_pika")
    pika_right = PikaController("right_pika")

    pika_left.set_up("/pika_pose_l","/gripper_l/joint_states")
    pika_right.set_up("/pika_pose_r","/gripper_r/joint_states")

    pika_left.set_collect_info(["end_pose","gripper"])
    pika_right.set_collect_info(["end_pose","gripper"])

    rospy.init_node('ros_subscriber_node', anonymous=True)

    while True:
        print("left_pika:\n", pika_left.get_state())
        print("right_pika:\n", pika_right.get_state())
        time.sleep(0.1)