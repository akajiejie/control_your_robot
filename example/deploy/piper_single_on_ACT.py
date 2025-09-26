import sys

sys.path.append("./")
import os
import numpy as np
import torch
import math
from my_robot.agilex_piper_single_base import PiperSingle
from policy.ACT.inference_model import MYACT
from utils.data_handler import is_enter_pressed
from data.collect_any import CollectAny
import time
import pdb
condition = {
    "save_path": "./test/", 
    "task_name": "pick_place_cup", 
    "save_format": "hdf5", 
    "save_freq": 50,
    "collect_type": "teleop",
}
def input_transform(data):
    has_left_arm = "left_arm" in data[0]
    has_right_arm = "right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": [0.0] * left_gripper_dim
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["left_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": [0.0] * right_gripper_dim
        }
        has_left_arm = True
    
    elif not has_left_arm and not has_right_arm:
        default_joint_dim = 6
        
        data[0]["left_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        data[0]["right_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        has_left_arm = True
        has_right_arm = True
    
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    ])
    # print(state)
    img_arr = data[1]["cam_head"]["color"], data[1]["cam_wrist"]["color"]
    return img_arr, state

def output_transform(data):
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
    left_joints = [
        clamp(data[0][i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    if data[0][6] < 0.05:
        data[0][6] = 0.0
    left_gripper = data[0][6]
    
    move_data = {
        "left_arm":{
            "joint": left_joints,
            "gripper": left_gripper,
        }
    }
    return move_data

if __name__ == "__main__":
    os.environ["INFO_LEVEL"] = "INFO"
    robot = PiperSingle()
    robot.set_up()
    collection=CollectAny(condition=condition,start_episode=0,move_check=True,resume=True)
    #load model
    model = MYACT("/home/usst/kwj/GitCode/control_your_robot_jie/policy/ACT/act_ckpt/act-pick_place_cup/100","act-pick_place_cup")
    max_step = 1200
    num_episode = 1
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
            action = model.get_action()
            move_data = output_transform(action)
            robot.move({"arm": 
                            move_data
                        })
            step += 1
            data = robot.get()
            collection.collect(data[0],None)
            # pdb.set_trace()
            time.sleep(1/robot.condition["save_freq"])
            print(f"Episode {i}, Step {step}/{max_step} completed.")

        robot.reset()
        collection.write()
        print("finish episode", i)
    robot.reset()


