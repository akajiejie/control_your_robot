import sys
sys.path.append('./')

# from my_robot.test_robot import TestRobot
from my_robot.agilex_piper_single_base import PiperSingle
from utils.bisocket import BiSocket
from utils.data_handler import debug_print, is_enter_pressed, hdf5_groups_to_dict
from my_robot.base_robot import dict_to_list
from policy.ACT.inference_model import MYACT
import socket
import time
import numpy as np
import math
import cv2
import pdb
import h5py
import os
class Replay:
    def __init__(self, hdf5_path) -> None:
        self.ptr = 0
        self.episode = dict_to_list(hdf5_groups_to_dict(hdf5_path))
    def get_data(self):
        # print(self.episode[self.ptr].keys())
        data = self.episode[self.ptr], self.episode[self.ptr]
        self.ptr += 1 #for act step is 1
        return data

class ActionCollector:
    """专门用于收集和保存模型输出action的类，参考collect_any实现"""
    def __init__(self, save_path="save/model_actions", task_name="act_inference"):
        self.episode = []
        self.episode_index = 0
        self.save_path = save_path
        self.task_name = task_name
        
        # 确保保存目录存在
        full_save_path = os.path.join(save_path, task_name)
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path)
    
    def collect(self, action, state, step):
        """收集一个时间步的数据（只保存动作和状态，不保存图像）"""
        episode_data = {
            "step": step,
            "action": np.array(action) if not isinstance(action, np.ndarray) else action,
            "state": np.array(state) if not isinstance(state, np.ndarray) else state,
            "timestamp": time.time()
        }
        self.episode.append(episode_data)
    
    def save_episode(self, custom_episode_name=None):
        """保存当前episode到hdf5文件
        
        Args:
            custom_episode_name: 可选的自定义episode名称，如果提供则使用该名称而不是episode_index
        """
        if len(self.episode) == 0:
            print("No data to save!")
            return
            
        full_save_path = os.path.join(self.save_path, self.task_name)
        
        # 如果提供了自定义名称，使用自定义名称；否则使用episode_index
        if custom_episode_name is not None:
            hdf5_path = os.path.join(full_save_path, f"{custom_episode_name}.hdf5")
        else:
            hdf5_path = os.path.join(full_save_path, f"episode_{self.episode_index}.hdf5")
        
        with h5py.File(hdf5_path, "w") as f:
            # 创建action组
            action_group = f.create_group("action")
            actions = np.array([ep["action"] for ep in self.episode])
            action_group.create_dataset("model_output", data=actions)
            
            # 创建observation组
            obs_group = f.create_group("observations")
            states = np.array([ep["state"] for ep in self.episode])
            obs_group.create_dataset("qpos", data=states)
            
            # 保存元数据
            metadata_group = f.create_group("metadata")
            steps = np.array([ep["step"] for ep in self.episode])
            timestamps = np.array([ep["timestamp"] for ep in self.episode])
            metadata_group.create_dataset("steps", data=steps)
            metadata_group.create_dataset("timestamps", data=timestamps)
        
        episode_name = custom_episode_name if custom_episode_name else f"episode_{self.episode_index}"
        print(f"Episode {episode_name} saved to {hdf5_path}")
        print(f"Total steps: {len(self.episode)}")
        
        # 重置episode并增加索引
        self.episode = []
        if custom_episode_name is None:  # 只有在使用默认命名时才增加索引
            self.episode_index += 1
    
    def reset(self):
        """重置收集器"""
        self.episode = []
    
def input_transform(data):
    has_left_arm = "slave_left_arm" in data[0]
    has_right_arm = "slave_right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["slave_left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["slave_right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": [0.0] * left_gripper_dim
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["slave_right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["slave_left_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": [0.0] * right_gripper_dim
        }
        has_left_arm = True
    
    elif not has_left_arm and not has_right_arm:
        default_joint_dim = 6
        
        data[0]["slave_left_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        data[0]["slave_right_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        has_left_arm = True
        has_right_arm = True
    
    state = np.concatenate([
        np.array(data[0]["slave_left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["slave_left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["slave_right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["slave_right_arm"]["gripper"]).reshape(-1)
    ])
    # print(state)
    img_arr = data[1]["slave_cam_head"]["color"], data[1]["slave_cam_wrist"]["color"]
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

class Client:
    def __init__(self,robot,cntrol_freq=10):
        self.robot = robot
        self.cntrol_freq = cntrol_freq
    
    def set_up(self, bisocket:BiSocket):
        self.bisocket = bisocket

    def move(self, message):
        # print(message)
        action_chunk = message["action_chunk"]
        action_chunk = np.array(action_chunk)

        for action in action_chunk[:30]:
            move_data = output_transform(action)
            # if move_data["arm"]["left_arm"]["gripper"] < 0.2 :
            #     move_data["arm"]["left_arm"]["gripper"] = 0.0
            # if move_data["arm"]["right_arm"]["gripper"] < 0.2 :
            #     move_data["arm"]["right_arm"]["gripper"] = 0.0
            
            self.robot.move(move_data)
            time.sleep(0.1)

    def play_once(self, R=None):
        if R:
            raw_data = R.get_data()
        else:    
            time.sleep(0.1)
            raw_data = self.robot.get()
        # print(raw_data)
            
        img_arr, state = input_transform(raw_data)
        print(state)
        data_send = {
            "img_arr": img_arr,
            "state": state
        }

        # send data
        self.bisocket.send(data_send)

        time.sleep(1 / self.cntrol_freq)

    def close(self):
        return

if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    robot = PiperSingle()
    robot.set_up()
    model = MYACT("/home/usst/kwj/GitCode/control_your_robot_jie/policy/ACT/act_ckpt/act-pick_place_cup/100","act-pick_place_cup")
    robot.reset()
    time.sleep(1)


    # 源文件夹路径
    source_folder = "save/pick_place_cup/"
    
    # 获取文件夹下所有hdf5文件
    hdf5_files = []
    for filename in os.listdir(source_folder):
        if filename.endswith('.hdf5'):
            hdf5_files.append(os.path.join(source_folder, filename))
    
    # 按文件名排序，确保按顺序处理
    hdf5_files.sort()
    
    if not hdf5_files:
        print(f"在文件夹 {source_folder} 中没有找到hdf5文件")
        exit()
    
    print(f"找到 {len(hdf5_files)} 个hdf5文件:")
    for file in hdf5_files:
        print(f"  - {file}")
    
    # 创建action收集器
    action_collector = ActionCollector(save_path="test/pick_place_cup/model_actions/", task_name="act_pick_place_cup")
    
    # 遍历每个hdf5文件进行测试
    for file_idx, source_hdf5_path in enumerate(hdf5_files):
        print(f"\n开始处理第 {file_idx + 1}/{len(hdf5_files)} 个文件: {source_hdf5_path}")
        
        re = Replay(source_hdf5_path)
        max_step = len(re.episode)  # 使用hdf5文件的实际步数
        print(f"HDF5文件包含 {max_step} 步数据")
        
        # 从源文件路径中提取文件名（不含扩展名）
        source_filename = os.path.splitext(os.path.basename(source_hdf5_path))[0]
        print(f"源文件名: {source_filename}")
        
        # 处理当前文件
        step = 0
        # 重置所有信息
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()
        action_collector.reset()  # 重置action收集器
        
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
            # data = robot.get()
            # print(data[0])
            raw_data=re.get_data()
            # print(raw_data[0])
            # data[0]["left_arm"] = raw_data[0]["left_arm"]
            img_arr, state = input_transform(raw_data)
            
            model.update_observation_window(img_arr, state)
            action = model.get_action()
            
            # 收集模型输出的action数据（只保存动作和状态）
            action_collector.collect(action, state, step)
            
            move_data = output_transform(action)
            # robot.move({"arm": 
            #                 move_data
            #             })
            step += 1
            # data = robot.get()
            # pdb.set_trace()
            time.sleep(1/robot.condition["save_freq"])
            print(f"File {file_idx + 1}/{len(hdf5_files)}, Step {step}/{max_step} completed. Action saved.")
        
        # Episode完成，保存收集的action数据，使用源文件名作为episode名称
        action_collector.save_episode(custom_episode_name=f"episode_{source_filename}")
        robot.reset()
        print(f"完成处理文件 {source_filename}")
    
    print(f"\n所有文件处理完成！总共处理了 {len(hdf5_files)} 个文件")
    robot.reset()
