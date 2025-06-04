import sys
sys.path.append("./")
import os

from my_robot.test_robot import TestRobot
from my_robot.agilex_piper_dual import PiperDual
import time
import keyboard
import numpy as np 
import math
from policy.RDT.inference_model import RDT
import pdb
from utils.data_handler import is_enter_pressed
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import cv2
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
joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),    # joint2
        (math.radians(-170), math.radians(0)),   # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),   # joint5
        (math.radians(-120), math.radians(120))    # joint6
    ]
gripper_limit=[(0.00,0.07)]
def input_transform(data):
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    ])


    img_arr = data[1]["cam_head"]["color"], data[1]["cam_right_wrist"]["color"], data[1]["cam_left_wrist"]["color"]
    return img_arr, state

def output_transform(data):
    # 2. 安全限位处理函数
    def clamp(value, min_val, max_val):
        """将值限制在[min_val, max_val]范围内"""
        return max(min_val, min(value, max_val))
    left_joints = [
        clamp(data[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    left_gripper = clamp(data[6], gripper_limit[0][0], gripper_limit[0][1])
    
    # 4. 处理右臂数据
    right_joints = [
        clamp(data[i+7], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    right_gripper = clamp(data[13], gripper_limit[0][0], gripper_limit[0][1])
    
    # 5. 构建输出结构
    move_data = {
        "left_arm": {
            "joint": left_joints,
            "gripper": left_gripper
        },
        "right_arm": {
            "joint": right_joints,
            "gripper": right_gripper
        }
    }
    return move_data

class DataLogger:
    def __init__(self):
        self.records = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("execution_logs", exist_ok=True)
    
    def log(self, step, move_data):
        """记录每一步的执行数据"""
        record = {
            "step": step,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")
        }
        
        # 提取左臂数据
        for i, joint in enumerate(move_data["left_arm"]["joint"]):
            record[f"left_joint_{i+1}"] = math.degrees(joint)  # 转换为角度
        record["left_gripper"] = move_data["left_arm"]["gripper"]
        
        # 提取右臂数据
        for i, joint in enumerate(move_data["right_arm"]["joint"]):
            record[f"right_joint_{i+1}"] = math.degrees(joint)  # 转换为角度
        record["right_gripper"] = move_data["right_arm"]["gripper"]
        
        self.records.append(record)
    
    def save_to_csv(self, episode):
        """保存为CSV文件"""
        filename = f"execution_logs/episode_{episode}_{self.timestamp}.csv"
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f"✅ 执行数据已保存至: {os.path.abspath(filename)}")
        return df
    
    def visualize(self, df, episode):
        """生成可视化图表"""
        plt.figure(figsize=(15, 10))
        
        # 1. 关节角度变化趋势
        plt.subplot(2, 1, 1)
        for i in range(1, 7):
            plt.plot(df["step"], df[f"left_joint_{i}"], label=f"Left Joint {i}")
            plt.plot(df["step"], df[f"right_joint_{i}"], linestyle="--", label=f"Right Joint {i}")
        plt.title(f"Episode {episode} - 关节角度变化趋势")
        plt.xlabel("执行步数")
        plt.ylabel("角度 (°)")
        plt.legend(ncol=3, loc="upper right")
        plt.grid(True)
        
        # 2. 夹爪状态变化
        plt.subplot(2, 1, 2)
        plt.plot(df["step"], df["left_gripper"], "b-", label="左夹爪")
        plt.plot(df["step"], df["right_gripper"], "r-", label="右夹爪")
        plt.title("夹爪开合状态")
        plt.xlabel("执行步数")
        plt.ylabel("开合值 (m)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        img_path = f"execution_logs/episode_{episode}_{self.timestamp}.png"
        plt.savefig(img_path)
        print(f"📈 可视化图表已保存至: {os.path.abspath(img_path)}")
        plt.close()

if __name__ == "__main__":
    # logger = DataLogger()
    robot = PiperDual()
    robot.set_up()
    # load model
    model = RDT("output/RDT/wenlong/6.4_10w/mp_rank_00_model_states_10w.pt", "stack_plates")
    max_step = 1000
    num_episode = 10

    for i in range(num_episode):
        step = 0
        # 重置所有信息
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()
        
        # 等待允许执行推理指令, 按enter开始
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("start to inference...")
            else:
                print("waiting for start command...")
                time.sleep(1)

        # 开始逐条推理运行
        while step < max_step:
            data = robot.get()
            img_arr, state = input_transform(data)
            model.update_observation_window(img_arr, state)
            print("get_action")
            action_chunk = model.get_action()
            print("action_chunk", action_chunk.shape)
            for action in action_chunk:
                move_data = output_transform(action)
                # print(move_data)
                robot.move(move_data)
                # logger.log(step, move_data)
                step += 1
                time.sleep(1/robot.condition["save_interval"])

        # df = logger.save_to_csv(i)
        # logger.visualize(df, i)
        # logger.records = []
        robot.reset()
        print("finish episode", i)
    robot.reset()
    