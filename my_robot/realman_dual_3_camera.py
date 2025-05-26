import sys
sys.path.append("./")
from controller.Realman_controller import RealmanController
from sensor.Realsense_sensor import RealsenseSensor
from data.collect_any import CollectAny
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

import numpy as np

# 组装你的控制器
CAMERA_SERIALS = {
    'head': '111',  # Replace with actual serial number
    'left_wrist': '111',   # Replace with actual serial number
    'right_wrist': '111',   # Replace with actual serial number
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    65,   # Joint 1
    38,    # Joint 2
    -66,  # Joint 3
    12,   # Joint 4
    6,  # Joint 5
    119,    # Joint 6
    66    # Joint 7
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    -59,   # Joint 1
    38,    # Joint 2
    -123,  # Joint 3
    -9,   # Joint 4
    -10,  # Joint 5
    -120,    # Joint 6
    32    # Joint 7
]

'''
最终保存信息包含:
condition = {
    image_keys: list[str], # 视角名称 [front_image, left_wrist_image, right_wrist_image]
    arm_type: str, # 机械臂类型
    state_is_joint: bool, # 关节角度是否为state
    is_action: bool, # 是否包含action
    is_dual: bool, # 是否为双臂
    save_right_now: bool, # 是否在当前时刻保存数据
    save_depth: bool, # 是否保存深度图
    save_path: str, # 保存路径
    task_name: str, # 任务名称
    save_format: str, # 保存格式
    save_interval: int, # 保存频率
}
'''

# 记录统一的数据操作信息, 相关配置信息由CollectAny补充并保存
condition = {
    "save_path": "./save/", # 保存路径
    "task_name": "test", # 任务名称
    "save_interval": 10, # 保存频率
}

class MyRobot:
    def __init__(self, start_episode=0):
        self.arm_controllers = {
            "left_arm": RealmanController("left_arm"),
            "right_arm": RealmanController("right_arm"),
        }
        self.image_sensors = {
            "cam_head": RealsenseSensor("cam_head"),
            "cam_left_wrist": RealsenseSensor("cam_left_wrist"),
            "cam_right_wrist": RealsenseSensor("cam_right_wrist"),
        }
        self.condition = condition
        self.collection = CollectAny(condition, start_episode=0)

    def set_up(self):
        self.arm_controllers["left_arm"].set_up("192.168.80.18", rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.arm_controllers["right_arm"].set_up("192.168.80.19", rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.image_sensors["cam_head"].set_up(CAMERA_SERIALS['head'], is_depth=False)
        self.image_sensors["cam_left_wrist"].set_up(CAMERA_SERIALS['left_wrist'], is_depth=False)
        self.image_sensors["cam_right_wrist"].set_up(CAMERA_SERIALS['right_wrist'], is_depth=False)
        self.set_collect_type(["joint","qpos","gripper"],["color"])
        print("set up success!")
        
    def get(self):
        controller_data = {}
        if self.arm_controllers is not None:    
            for controller_name, controller in self.arm_controllers.items():
                controller_data[controller_name] = controller.get()
        sensor_data = {}
        if self.image_sensors is not None:  
            for sensor_name, sensor in self.image_sensors.items():
                sensor_data[sensor_name] = sensor.get()
        return [controller_data, sensor_data]
    
    def move(self, move_data):
        self.arm_controllers["left_arm"].move(move_data["left_arm"],is_delta=False)
        self.arm_controllers["right_arm"].move(move_data["right_arm"],is_delta=False)
    
    def reset(self):
        self.arm_controllers["left_arm"].reset(START_POSITION_ANGLE_LEFT_ARM)
        self.arm_controllers["right_arm"].reset(START_POSITION_ANGLE_RIGHT_ARM)

    def finish(self):
        self.collection.write()
    
    def set_collect_type(self,ARM_INFO_NAME,IMG_INFO_NAME):
        for controller in self.arm_controllers.values():
            controller.set_collect_info(ARM_INFO_NAME)
        for sensor in self.image_sensors.values():
            sensor.set_collect_info(IMG_INFO_NAME)

    def is_start(self):
        if max(abs(self.arm_controllers["left_arm"].get_state()["joint"] - START_POSITION_ANGLE_LEFT_ARM), abs(self.arm_controllers["right_arm"].get_state()["joint"] - START_POSITION_ANGLE_RIGHT_ARM)) > 0.01:
            return True
        else:
            return False
    
    def collect(self, data):
        self.collection.collect(data[0], data[1])
    
    def set_action(self, action):
        self.arm_controllers["left_arm"].set_action(action)
        self.arm_controllers["right_arm"].set_action(action)

# 编写你的机器人测试样例
if __name__ == "__main__":
    robot = MyRobot()
    # 初始化机器人
    robot.reset()
    '''
    测试接口:
    1. 机械臂位置初始化
    2. 机械臂是否运动判定
    3. 机械臂获取数据
    4. 机械臂运动
    '''
    #
    print(robot.is_start())

    print(robot.get())


    