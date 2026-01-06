import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.Piper_controller import PiperController
from sensor.Realsense_sensor import RealsenseSensor
from sensor.Vitac3D import Vitac3D
from data.collect_any import CollectAny

# setting your realsense serial
CAMERA_SERIALS = {
    'head': '337122301391',  # Replace with actual serial number
    'left_wrist': '420122070816',   # Replace with actual serial number
    'right_wrist': '338622074268',   # Replace with actual serial number
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
    0.0,   # Joint 1
    0.85220935,    # Joint 2
    -0.68542569,  # Joint 3
    0.,   # Joint 4
    0.78588684,  # Joint 5
    0.0,    # Joint 6
]

condition = {
    "save_path": "./save/",
    "task_name": "test",
    "save_format": "hdf5",
    "save_freq": 10, 
}

class PiperDual(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.controllers = {
            "arm":{
                "left_arm": PiperController("left_arm"),
                "right_arm": PiperController("right_arm"),
            }
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_left_wrist": RealsenseSensor("cam_left_wrist"),
                "cam_right_wrist": RealsenseSensor("cam_right_wrist"),
            },
            "tactile":{
                "right_arm_left_tac": Vitac3D("right_left_tac"),
                "right_arm_right_tac": Vitac3D("right_right_tac"),
                "left_arm_left_tac": Vitac3D("left_left_tac"),
                "left_arm_right_tac": Vitac3D("left_right_tac"),
            },
        }

    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))
        self.controllers["arm"]["right_arm"].reset(np.array(START_POSITION_ANGLE_RIGHT_ARM))

    def set_up(self):
        super().set_up()
        self.controllers["arm"]["left_arm"].set_up("can_left")
        self.controllers["arm"]["right_arm"].set_up("can_right")

        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS['head'], is_depth=False)
        self.sensors["image"]["cam_left_wrist"].set_up(CAMERA_SERIALS['left_wrist'], is_depth=False)
        self.sensors["image"]["cam_right_wrist"].set_up(CAMERA_SERIALS['right_wrist'], is_depth=False)
        self.sensors["tactile"]["right_arm_left_tac"].set_up("/dev/ttyUSB0",is_show=False)
        self.sensors["tactile"]["right_arm_right_tac"].set_up("/dev/ttyUSB1",is_show=False)
        self.sensors["tactile"]["left_arm_left_tac"].set_up("/dev/ttyUSB2",is_show=False)
        self.sensors["tactile"]["left_arm_right_tac"].set_up("/dev/ttyUSB3",is_show=False)

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"],
                               "tactile": ["tactile"]
                               })
        print("set up success!")

if __name__ == "__main__":
    import time
    
    robot = PiperDual()
    robot.set_up()

    # collection test
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    
    robot.reset()
    # moving test
    # move_data = {
    #     "arm":{
    #         "left_arm":{
    #         "joint":[
    #                 0.0,   # Joint 1
    #                 0.85220935,    # Joint 2
    #                 -0.68542569,  # Joint 3
    #                 0.,   # Joint 4
    #                 0.78588684,  # Joint 5
    #                 0.0,    # Joint 6
    #                 ],
    #         "gripper":0.2,
    #         },
    #         "right_arm":{
    #         "joint":[
    #                 0.0,   # Joint 1
    #                 0.85220935,    # Joint 2
    #                 -0.68542569,  # Joint 3
    #                 0.,   # Joint 4
    #                 0.78588684,  # Joint 5
    #                 0.0,    # Joint 6
    #                 ],
    #         "gripper":0.2,
    #         },
    #     }
    # }
    # robot.move(move_data)
    
    # move_data = {
    #     "arm":{
    #         "right_arm":{
    #         "joint":[
    #                 0.0,   # Joint 1
    #                 0.85220935,    # Joint 2
    #                 -0.68542569,  # Joint 3
    #                 0.,   # Joint 4
    #                 0.78588684,  # Joint 5
    #                 0.0,    # Joint 6
    #                 ],
    #         "gripper":0.2,
    #         },
    #     }
    # }
    # robot.move(move_data)
