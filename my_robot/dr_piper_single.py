
import numpy as np

from controller.Piper_controller import PiperController
from sensor.Reaksense_MultiThread_sensor import RealsenseSensor
from sensor.Vitac3D import Mutilthread3DVitac
from data.collect_any import CollectAny
from controller.drAloha_controller import DrAlohaController
from typing import Dict, Any
import math
import time
# 组装你的控制器
CAMERA_SERIALS = {
    'head': '420122070816',  # Replace with actual serial number
    'left_wrist': '948122073452',   # Replace with actual serial number
    # 'right_wrist': '1111',   # Replace with actual serial number
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
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
    "save_path": "./save/", # 保存路径
    "task_name": "test", # 任务名称
    "save_format": "hdf5", # 保存格式
    "save_interval": 30, # 保存频率
}

class dr_Single:
    def __init__(self, start_episode=0):
        self.master_arm_controllers = {
            "left_arm": DrAlohaController("left_master_arm"),
        }
        self.arm_controllers = {
            "left_arm": PiperController("left_arm"),
        }
        self.image_sensors = {
            "cam_head": RealsenseSensor("cam_head"),
            "cam_wrist": RealsenseSensor("cam_wrist"),
        }
        self.tactile_sensor = {
            "left_arm_left": Mutilthread3DVitac("left_arm_left"),
        }
        self.condition = condition
        self.collection = CollectAny(condition, start_episode=0)
    #============== 初始化相关 ==============
    def set_up(self):
        self.master_arm_controllers["left_arm"].set_up("/dev/ttyACM0")
        self.arm_controllers["left_arm"].set_up("can_left")
        self.image_sensors["cam_head"].set_up(CAMERA_SERIALS["head"])
        self.image_sensors["cam_wrist"].set_up(CAMERA_SERIALS["left_wrist"])
        self.tactile_sensor["left_arm_left"].set_up("/dev/ttyUSB0",is_show=False)
        self.set_collect_type(["joint","gripper"],["color"],["force"])
        self.image_sensors["cam_head"]._start_thread()  # 启动摄像头线程
        self.image_sensors["cam_wrist"]._start_thread()  # 启动
        
        #先不采集qpos
        print("set up success!")
    
    def reset(self):
        self.arm_controllers["left_arm"].reset(START_POSITION_ANGLE_LEFT_ARM)
    
    def set_collect_type(self,ARM_INFO_NAME,IMG_INFO_NAME,TAC_INFO_NAME):
        for sensor in self.image_sensors.values():
            sensor.set_collect_info(IMG_INFO_NAME)
        for tactile in self.tactile_sensor.values():
            tactile.set_collect_info(TAC_INFO_NAME)
        for controller in self.arm_controllers.values():
            controller.set_collect_info(ARM_INFO_NAME)
        
        
    # ============== 机械臂判定相关 ============== 
    def is_start(self):
        return True
        if abs(self.arm_controllers["left_arm"].get_state()["joint"] - np.array(START_POSITION_ANGLE_LEFT_ARM)) > 0.01:
            return True
        else:
            return False
    # ============== 数据操作相关 ==============
    def get(self):
        controller_data = {}
        if self.arm_controllers is not None:    
            for controller_name, controller in self.arm_controllers.items():
                controller_data[controller_name] = controller.get()
        sensor_data = {}
        if self.image_sensors is not None:  
            for sensor_name, sensor in self.image_sensors.items():
                sensor_data[sensor_name] = sensor.get()
        if self.tactile_sensor is not None:
            for tactile_name, tactile in self.tactile_sensor.items():
                sensor_data[tactile_name] = tactile.get_touch()

        return [controller_data, sensor_data]
    
    def collect(self, data):
        self.collection.collect(data[0], data[1])
    
    def finish(self):
        self.collection.write()
    # ============== 运动操作相关 ==============
    def set_action(self, action):
        pass
    
    def move(self, move_data):  
        self.arm_controllers["left_arm"].move(move_data["left_arm"],is_delta=False)
    def action_transform(self, move_data:Dict[str, Any]):
        """ Transform the action from master arm to the slave arm."""
        joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),    # joint2
        (math.radians(-170), math.radians(0)),   # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),   # joint5
        (math.radians(-120), math.radians(120))    # joint6
        ]

        def clamp(value, min_val, max_val):
            """将值限制在[min_val, max_val]范围内"""
            return max(min_val, min(value, max_val))
         # 关键修改1：提取关节数据并转换为列表
        joint_data_dict = move_data["joint"].item()
        joints = [joint_data_dict[i] for i in range(6)]  # 转换为索引访问的列表
        
        # 关键修改2：关节角度校准（度→弧度转换后调整）
        joints[1] = joints[1] - math.radians(90)   # 关节2校准：减去90°
        joints[2] = joints[2] + math.radians(175)  # 关节3校准：增加175°
        
        # 关键修改3：关节方向反转（特定关节取负）
        for i in [1, 2, 4]:  # 关节2、3、5需反转方向
            joints[i] = -joints[i]
        left_joints = [
            clamp(joints[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
        ]
        # gripper_data = move_data["gripper"]
        left_gripper = move_data["gripper"]
        
        # 4. 处理右臂数据
        # right_joints = [
        #     clamp(move_data[i+7], joint_limits_rad[i][0], joint_limits_rad[i][1])
        #     for i in range(6)
        # ]
        # right_gripper = clamp(move_data[13], gripper_limit[0][0], gripper_limit[0][1])
        # right_gripper = right_gripper * 1000 / 70
        # 5. 构建输出结构
        move_data = {
            "left_arm": {
                "joint": left_joints,
                "gripper": left_gripper
            }
            # "right_arm": {
            #     "joint": right_joints,
            #     "gripper": right_gripper
            # }
        }
        return move_data
    def teleoperation_setp(self,is_record=False,force_feedback=False):
        
        left_master_action=self.master_arm_controllers["left_arm"].get_state()
        action=self.action_transform(left_master_action)
        return action
if __name__ == "__main__":
    robot = dr_Single()
    robot.set_up()
    robot.master_arm_controllers["left_arm"].zero_gravity()
    # 采集测试
    data_list = []
    for i in range(300):
        print(i)
        action=robot.teleoperation_setp()
        robot.move(action)
        data = robot.get()
        robot.collect(data)
        time.sleep(1/robot.condition["save_interval"])
    robot.finish()