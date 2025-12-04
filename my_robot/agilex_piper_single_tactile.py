import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.Piper_controller import PiperController
from sensor.Realsense_sensor import RealsenseSensor
from sensor.Vitac3D import Vitac3D
# from sensor.Realsense_MultiThread_sensor import RealsenseSensor

from data.collect_any import CollectAny

CAMERA_SERIALS = {
    'head': '420122070816',  # Replace with actual serial number
    'wrist': '338622074268',   # Replace with actual serial number
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0.0,   # Joint 1
    0.85220935,    # Joint 2
    -0.68542569,  # Joint 3
    0.,   # Joint 4
    0.78588684,  # Joint 5
    0.0,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "robot":"piper_single",
    "save_path": "./datasets/", 
    "task_name": "test", 
    "save_format": "hdf5", 
    "save_freq": 30, 
}


class PiperSingle(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.condition = condition
        self.controllers = {
            "arm":{
                "right_arm": PiperController("right_arm"),
            },
        }
        self.sensors = {
            "image":{
                "cam_head": RealsenseSensor("cam_head"),
                "cam_right_wrist": RealsenseSensor("cam_right_wrist"),
            },
            "tactile":{
                "right_arm_tac": Vitac3D("right_arm_tac"),
            },
        }

    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["right_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))

    def set_up(self):
        super().set_up()

        self.controllers["arm"]["right_arm"].set_up("can0")
        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS["head"])
        self.sensors["image"]["cam_right_wrist"].set_up(CAMERA_SERIALS["wrist"])
        self.sensors["tactile"]["right_arm_tac"].set_up("/dev/ttyUSB0",is_show=False)

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"],
                               "tactile": ["tactile"]
                               })
        
        print("set up success!")
    
if __name__=="__main__":
    import time
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    robot = PiperSingle()
    robot.set_up()
    
    # collection test
    robot.reset()
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    
    # # moving test
    # move_data = {
    #     "arm":{
    #         "left_arm":{
    #         "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
    #         "gripper":0.2,
    #         },
    #     },
    # }
    # robot.move(move_data)
    # time.sleep(1)
    # move_data = {
    #     "arm":{
    #         "left_arm":{
    #         "joint":[0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "gripper":0.2,
    #         },
    #     },
    # }
    # robot.move(move_data)