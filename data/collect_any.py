'''
使用本函数将会一股脑存储数据, 不会按照任何条件进行筛选
存储的格式和双臂机械臂的规定格式不同,一个controller / sensor 对应一个group
'''
import os
import numpy as np
import h5py
import json


class CollectAny:
    def __init__(self, condition=None, start_episode=0):
        self.condition = condition
        self.episode = []
        self.episode_index = start_episode
    
    def collect(self, controllers_data, sensors_data):
        episode_data = {}
        if controllers_data is not None:    
            for controller_name, controller_data in controllers_data.items():
                episode_data[controller_name] = controller_data
        if sensors_data is not None:    
            for sensor_name, sensor_data in sensors_data.items():
                episode_data[sensor_name] = sensor_data
        self.episode.append(episode_data)
    
    def get_item(self, controller_name, item):
        if item in self.episode[0][controller_name]:
            return np.array([self.episode[i][controller_name][item] for i in range(len(self.episode))])
        else:
            print(f"item {item} not in {controller_name}")
            return None
        
    def write(self, only_end=False):
        save_path = os.path.join(self.condition["save_path"], f"{self.condition['task_name']}/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 判断是否存在config.json, 不存在则创建并写入
        condition_path = os.path.join(save_path, "./config.json")
        if not os.path.exists(condition_path):
             for key in self.episode[0].keys():
                 self.condition[key] = list(self.episode[0][key].keys())
             with open(condition_path, 'w', encoding='utf-8') as f:
                 json.dump(self.condition, f, ensure_ascii=False, indent=4)

        hdf5_path = os.path.join(save_path, f"{self.episode_index}.hdf5")
        with h5py.File(hdf5_path, "w") as f:
            obs = f
            for controller_name in self.episode[0].keys():
                controller_group = obs.create_group(controller_name)
                for item in self.episode[0][controller_name].keys():
                    data = self.get_item(controller_name, item)
                    controller_group.create_dataset(item, data=data)
        print(f"write to {hdf5_path}")
        # 清空当前的episode, 开始新的episode
        self.episode = []
        self.episode_index += 1



    
    
