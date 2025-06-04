import h5py
from typing import *
from pathlib import Path
import numpy as np
import os
import fnmatch
import sys
import select

from scipy.spatial.transform import Rotation


def compute_rotate_matrix(pose):
    """将位姿 [x,y,z,roll,pitch,yaw] 转换为齐次变换矩阵 (XYZ欧拉角顺序)"""
    x, y, z, roll, pitch, yaw = pose

    R = Rotation.from_euler('XYZ', [roll, pitch, yaw]).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T

def compute_local_delta_pose(base_pose, target_pose):
    """
    计算局部坐标系下的位姿增量 (基于base_pose的坐标系)
    参数:
        base_pose: 基准位姿 [x,y,z,roll,pitch,yaw]
        target_pose: 目标位姿 [x,y,z,roll,pitch,yaw]
    返回:
        增量位姿 [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
    """
    assert len(base_pose) == 6 and len(target_pose) == 6, "输入位姿必须是6维"
    
    # 计算旋转增量
    base_rotate = Rotation.from_euler('XYZ', base_pose[3:])
    target_rotate = Rotation.from_euler('XYZ', target_pose[3:])
    delta_rotate = base_rotate.inv() * target_rotate
    delta_rpy = delta_rotate.as_euler('XYZ', degrees=False)
    
    # 计算平移增量（转换到局部坐标系）
    delta_global = np.array(target_pose[:3]) - np.array(base_pose[:3])
    delta_xyz = base_rotate.inv().apply(delta_global)
    
    return np.concatenate([delta_xyz, delta_rpy])


def get_item(Dict_data: Dict, item):
    if isinstance(item, str):
        keys = item.split(".")
        data = Dict_data
        for key in keys:
            data = data[key]
    elif isinstance(item, list):
        key_item = None
        for it in item:
            now_data = get_item(Dict_data, it)
            # import pdb;pdb.set_trace()
            if key_item is None:
                key_item = now_data
            else:
                key_item = np.column_stack((key_item, now_data))
        data = key_item
    else:
        raise ValueError(f"input type is not allow!")
    return data

def hdf5_groups_to_dict(hdf5_path):
    """
    读取HDF5文件中所有group，并转换为嵌套字典结构
    
    参数:
        hdf5_path: HDF5文件路径
        
    返回:
        包含所有group数据的嵌套字典
    """
    result = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # 遍历文件中的所有对象
        def visit_handler(name, obj):
            if isinstance(obj, h5py.Group):
                group_dict = {}
                # 遍历group中的所有数据集
                for key in obj.keys():
                    if isinstance(obj[key], h5py.Dataset):
                        group_dict[key] = obj[key][()]
                result[name] = group_dict
                
        f.visititems(visit_handler)
    
    return result

def get_files(directory, extension):
    """使用pathlib获取所有匹配的文件"""
    file_paths = []
    for root, _, files in os.walk(directory):
            for filename in fnmatch.filter(files, extension):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    return file_paths

def debug_print(name, info, level="INFO"):
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    if level not in levels.keys():
        debug_print("DEBUG_PRINT", f"level setting error : {level}", "ERROR")
        return
    env_level = os.getenv("INFO_LEVEL", "INFO").upper()
    env_level_value = levels.get(env_level, 20)

    msg_level_value = levels.get(level.upper(), 20)

    if msg_level_value < env_level_value:
        return

    colors = {
        "DEBUG": "\033[94m",   # blue
        "INFO": "\033[92m",    # green
        "WARNING": "\033[93m", # yellow
        "ERROR": "\033[91m",   # red
        "ENDC": "\033[0m",
    }
    color = colors.get(level.upper(), "")
    endc = colors["ENDC"]
    print(f"{color}[{level}][{name}] {info}{endc}")

def is_enter_pressed():
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == '\n'    

if __name__=="__main__":
    s = np.array([0.1, 0.2, 0.3, 0., 0., 0.1])
    print(compute_rotate_matrix(s))

    base_pose = np.array([0, 0, 0, 0, 0, 1])
    target_pose = np.array([1, 1, 1, 1, 0, 0])
    print(compute_local_delta_pose(base_pose, target_pose))