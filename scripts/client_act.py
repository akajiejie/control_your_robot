import sys
sys.path.append('./')

# from my_robot.test_robot import TestRobot
from my_robot.agilex_piper_single_base import PiperSingle
from utils.bisocket import BiSocket
from utils.data_handler import debug_print, is_enter_pressed, hdf5_groups_to_dict
from my_robot.base_robot import dict_to_list
from data.collect_any import CollectAny
from policy.ACT.inference_model import MYACT
import socket
import time
import numpy as np
import math
import cv2
import pdb
import h5py
import os

def decode_compressed_images(encoded_data):
    """解码压缩的JPEG图像数据
    
    Args:
        encoded_data: 压缩的JPEG字节数据数组
        
    Returns:
        解码后的图像数组
    """
    imgs = []
    for data in encoded_data:
        # 移除填充的零字节
        jpeg_bytes = data.tobytes().rstrip(b"\0")
        # 将字节数据转换为numpy数组
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        # 解码JPEG图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
        else:
            print("警告：无法解码图像数据")
    return np.array(imgs)
condition = {
    "save_path": "./test/reload_model_actions/", 
    "task_name": "act_stack_bowls_two", 
    "save_format": "hdf5", 
    "save_freq": 50,
    "collect_type": "teleop",
}
class Replay:
    def __init__(self, hdf5_path, is_compressed=False) -> None:
        self.ptr = 0
        self.is_compressed = is_compressed
        
        if is_compressed:
            # 处理压缩格式的数据
            self.episode_data = self._load_compressed_data(hdf5_path)
        else:
            # 原始格式
            self.episode = dict_to_list(hdf5_groups_to_dict(hdf5_path))
    
    def _load_compressed_data(self, hdf5_path):
        """加载压缩格式的HDF5数据"""
        episode_data = []
        
        with h5py.File(hdf5_path, 'r') as f:
            obs = f['observations']
            
            # 解码压缩的图像数据 - 支持三相机
            cam_high_encoded = obs['cam_high'][:]
            cam_high_images = decode_compressed_images(cam_high_encoded)
            
            # 检查是否有左右手腕相机
            cam_left_wrist_images = None
            cam_right_wrist_images = None
            cam_wrist_images = None
            
            if 'cam_left_wrist' in obs and 'cam_right_wrist' in obs:
                # 双臂配置：左右手腕相机
                cam_left_wrist_encoded = obs['cam_left_wrist'][:]
                cam_right_wrist_encoded = obs['cam_right_wrist'][:]
                cam_left_wrist_images = decode_compressed_images(cam_left_wrist_encoded)
                cam_right_wrist_images = decode_compressed_images(cam_right_wrist_encoded)
            elif 'cam_wrist' in obs:
                # 单臂配置：单个手腕相机
                cam_wrist_encoded = obs['cam_wrist'][:]
                cam_wrist_images = decode_compressed_images(cam_wrist_encoded)
            
            # 获取机械臂数据
            left_arm_joint = obs['left_arm']['joint'][:]
            left_arm_gripper = obs['left_arm']['gripper'][:]
            
            # 检查是否有右臂数据
            right_arm_joint = None
            right_arm_gripper = None
            if 'right_arm' in obs and len(obs['right_arm'].keys()) > 0:
                if 'joint' in obs['right_arm'] and 'gripper' in obs['right_arm']:
                    right_arm_joint = obs['right_arm']['joint'][:]
                    right_arm_gripper = obs['right_arm']['gripper'][:]
            
            # 构建episode数据
            for i in range(len(left_arm_joint)):
                step_data = {
                    'left_arm': {
                        'joint': left_arm_joint[i],
                        'gripper': left_arm_gripper[i]
                    },
                    'cam_high': {
                        'color': cam_high_images[i] if i < len(cam_high_images) else cam_high_images[-1]
                    }
                }
                
                # 添加右臂数据（如果存在）
                if right_arm_joint is not None and right_arm_gripper is not None:
                    step_data['right_arm'] = {
                        'joint': right_arm_joint[i],
                        'gripper': right_arm_gripper[i]
                    }
                
                # 添加相机数据
                if cam_left_wrist_images is not None and cam_right_wrist_images is not None:
                    # 双臂配置
                    step_data['cam_left_wrist'] = {
                        'color': cam_left_wrist_images[i] if i < len(cam_left_wrist_images) else cam_left_wrist_images[-1]
                    }
                    step_data['cam_right_wrist'] = {
                        'color': cam_right_wrist_images[i] if i < len(cam_right_wrist_images) else cam_right_wrist_images[-1]
                    }
                elif cam_wrist_images is not None:
                    # 单臂配置
                    step_data['cam_wrist'] = {
                        'color': cam_wrist_images[i] if i < len(cam_wrist_images) else cam_wrist_images[-1]
                    }
                
                episode_data.append(step_data)
        
        return episode_data
    
    def get_data(self):
        if self.is_compressed:
            # 压缩格式：返回当前步骤的数据
            if self.ptr >= len(self.episode_data):
                return None, None
            
            step_data = self.episode_data[self.ptr]
            
            # 构建机械臂数据
            arm_data = {
                'left_arm': step_data['left_arm']
            }
            
            # 添加右臂数据（如果存在）
            if 'right_arm' in step_data:
                arm_data['right_arm'] = step_data['right_arm']
            
            # 构建图像数据
            img_data = {
                'cam_high': step_data['cam_high']
            }
            
            # 添加相机数据（根据实际存在的相机）
            if 'cam_left_wrist' in step_data and 'cam_right_wrist' in step_data:
                # 双臂配置
                img_data['cam_left_wrist'] = step_data['cam_left_wrist']
                img_data['cam_right_wrist'] = step_data['cam_right_wrist']
            elif 'cam_wrist' in step_data:
                # 单臂配置
                img_data['cam_wrist'] = step_data['cam_wrist']
            
            self.ptr += 10
            return arm_data, img_data
        else:
            # 原始格式
            data = self.episode[self.ptr], self.episode[self.ptr]
            self.ptr += 10
            return data

    
def input_transform(data):
    has_left_arm = "left_arm" in data[0]
    has_right_arm = "right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": 0.0
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["left_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": 0.0
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
    
    # 处理图像数据 - 支持不同的相机配置
    if "cam_left_wrist" in data[1] and "cam_right_wrist" in data[1]:
        # 双臂配置：三相机
        img_arr = (
            data[1]["cam_high"]["color"], 
            data[1]["cam_left_wrist"]["color"],
            data[1]["cam_right_wrist"]["color"]
        )
    elif "cam_wrist" in data[1]:
        # 单臂配置：两相机
        img_arr = (
            data[1]["cam_high"]["color"], 
            data[1]["cam_wrist"]["color"]
        )
    else:
        # 只有头部相机
        img_arr = (data[1]["cam_high"]["color"],)
    
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
    right_joints = [
        clamp(data[0][i+7], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    if data[0][6] < 0.05:
        data[0][6] = 0.0
    left_gripper = data[0][6]
    if data[0][13] < 0.05:
        data[0][13] = 0.0
    right_gripper = data[0][13]
    move_data = {
        "left_arm":{
            "joint": left_joints,
            "gripper": left_gripper,
        },
        "right_arm":{
            "joint": right_joints,
            "gripper": right_gripper,
        }

    }
    return move_data

if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    model = MYACT("/home/usst/kwj/GitCode/control_your_robot_jie/policy/ACT/act_ckpt/stack_bowls_two","act-stack_bowls_two")
    collection=CollectAny(condition=condition,start_episode=0,move_check=True,resume=False)
    time.sleep(1)

    # 源文件夹路径 - 修改为压缩数据路径
    source_folder = "save/stack_bowls_two_zip/"
    
    # 获取文件夹下所有hdf5文件
    hdf5_files = []
    for filename in os.listdir(source_folder):
        if filename.endswith('.hdf5'):
            hdf5_files.append(os.path.join(source_folder, filename))
    
    # 按文件名中的数字顺序排序
    def extract_number(filename):
        """从文件名中提取数字用于排序"""
        import re
        basename = os.path.basename(filename)
        # 提取文件名中的数字部分
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[0])  # 使用第一个数字进行排序
        return 0  # 如果没有数字，返回0
    
    hdf5_files.sort(key=extract_number)
    
    if not hdf5_files:
        print(f"在文件夹 {source_folder} 中没有找到hdf5文件")
        exit()
    
    print(f"找到 {len(hdf5_files)} 个hdf5文件（按数字顺序排序）:")
    for i, file in enumerate(hdf5_files):
        basename = os.path.basename(file)
        print(f"  {i+1}. {basename} (完整路径: {file})")
    
    # 遍历每个hdf5文件进行测试
    for file_idx, source_hdf5_path in enumerate(hdf5_files):
        print(f"\n开始处理第 {file_idx + 1}/{len(hdf5_files)} 个文件: {source_hdf5_path}")
        
        re = Replay(source_hdf5_path, is_compressed=True)
        if re.is_compressed:
            max_step = len(re.episode_data)  # 压缩格式使用episode_data长度
        else:
            max_step = len(re.episode)  # 原始格式使用episode长度
        print(f"HDF5文件包含 {max_step} 步数据")
        
        # 从源文件路径中提取文件名（不含扩展名）
        source_filename = os.path.splitext(os.path.basename(source_hdf5_path))[0]
        print(f"源文件名: {source_filename}")
        
        # 处理当前文件
        step = 0
        # 重置所有信息
        model.reset_obsrvationwindows()
        model.random_set_language()
        # action_collector.reset()  # 重置action收集器
        
        # # 等待允许执行推理指令, 按enter开始
        # is_start = False
        # while not is_start:
        #     if is_enter_pressed():
        #         is_start = True
        #         print("start to inference...")
        #     else:
        #         print("waiting for start command...")
                # time.sleep(1)
        time.sleep(1)

        # 开始逐条推理运行
        while step < max_step:
            # print(data[0])
            raw_data=re.get_data()
            # print(raw_data[0])
            # data[0]["left_arm"] = raw_data[0]["left_arm"]
            img_arr, state = input_transform(raw_data)
            
            model.update_observation_window(img_arr, state)
            action_chunk = model.get_action() 
            action_chunk = action_chunk[:10]
            for action in action_chunk:
                # 将action数据转换为collect_any期望的格式
                controllers_data = {
                    "left_arm": {
                        "joint": action[:6].tolist(),  # 前6个是左臂关节角度
                        "gripper": action[6]  # 第7个是左臂夹爪状态
                    },
                    "right_arm": {
                        "joint": action[7:13].tolist(),  # 第8-13个是右臂关节角度
                        "gripper": action[13]  # 第14个是右臂夹爪状态
                    }
                }
                collection.collect(controllers_data, None)
                step += 1
                # pdb.set_trace()
                time.sleep(1/condition["save_freq"])
                print(f"File {file_idx + 1}/{len(hdf5_files)}, Step {step}/{max_step} completed. Action saved.")

        collection.write()
        time.sleep(1)
        print(f"完成处理文件 {source_filename}")
    
    print(f"\n所有文件处理完成！总共处理了 {len(hdf5_files)} 个文件")
