import sys
sys.path.append("./")

import os
from typing import Dict, Any, List
import numpy as np
import time
import json
import h5py

from controller.TestArm_controller import TestArmController
from sensor.TestVision_sensor import TestVisonSensor

from data.collect_any import CollectAny

from utils.data_handler import debug_print, hdf5_groups_to_dict


condition = {
    "save_path": "./save/", 
    "task_name": "base", 
    "save_format": "hdf5", 
    "save_freq": 10,
}

class Robot:
    def __init__(self, start_episode=0) -> None:
        self.name = "base_robot"
        self.controllers = {}
        self.sensors = {}

        self.condition = condition
        self.collection = CollectAny(condition, start_episode=start_episode)

    def set_up(self):
        debug_print(self.name, "set_up() should be realized by your robot class", "ERROR")
        raise NotImplementedError
    
    def set_collect_type(self,INFO_NAMES: Dict[str, Any]):
        for key,value in INFO_NAMES.items():
            if key in self.controllers:
                for controller in self.controllers[key].values():
                    controller.set_collect_info(value)
            if key in self.sensors:
                for sensor in self.sensors[key].values():
                    sensor.set_collect_info(value)
    
    def get(self):
        controller_data = {}
        sensor_data = {}

        if self.controllers is not None:
            for type_name, controller_type in self.controllers.items():
                for controller_name, controller in controller_type.items():
                    controller_data[controller_name] = controller.get()

        if self.sensors is not None:
            for type_name, sensor_type in self.sensors.items(): 
                for sensor_name, sensor in sensor_type.items():
                    sensor_data[sensor_name] = sensor.get()

        return [controller_data, sensor_data]
    
    def collect(self, data):
        self.collection.collect(data[0], data[1])
    
    def finish(self):
        self.collection.write()
    
    def move(self, move_data):
        if move_data is None:
            return
        for controller_type_name, controller_type in move_data.items():
            for controller_name, controller_action in controller_type.items():
                self.controllers[controller_type_name][controller_name].move(controller_action,is_delta=False)
    
    def is_start(self):
        debug_print(self.name, "your are using default func: is_start(), this will return True only", "DEBUG")
        return True

    def reset(self):
        debug_print(self.name, "your are using default func: reset(), this will return True only", "DEBUG")
        return True

    def replay(self, data_path, key_banned=None):
        parent_dir = os.path.dirname(data_path)
        config_path = os.path.join(parent_dir, "config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            condition = json.load(f)
        
        time_interval = 1.0 / condition['save_freq']

        episode = dict_to_list(hdf5_groups_to_dict(data_path))
        now_time = last_time = time.monotonic()
        for ep in episode:
            while now_time - last_time < time_interval:
                now_time = time.monotonic()
            self.play_once(ep)
            last_time = time.monotonic()
    
    def play_once(self, episode: Dict[str, Any]):
        for controller_group in self.controllers.values():
            for controller_name, controller in controller_group.items():
                if controller_name in episode:
                    controller.move(episode[controller_name], is_delta=False)

def get_array_length(data: Dict[str, Any]) -> int:
    """获取最外层np.array的长度"""
    for value in data.values():
        if isinstance(value, dict):
            return get_array_length(value)
        elif isinstance(value, np.ndarray):
            return value.shape[0]
    raise ValueError("No np.ndarray found in data.")

def split_nested_dict(data: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """提取每一帧的子结构"""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = split_nested_dict(value, idx)
        elif isinstance(value, np.ndarray):
            result[key] = value[idx]
        else:
            raise TypeError(f"Unsupported type: {type(value)} at key {key}")
    return result

def dict_to_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    length = get_array_length(data)
    return [split_nested_dict(data, i) for i in range(length)]