import h5py
from typing import *
from pathlib import Path
import numpy as np
import os
import fnmatch
import sys
import select

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