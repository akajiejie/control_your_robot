import sys
sys.path.append("./")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import json

from utils.data_handler import *

'''
设置map, 用于将默认数据格式映射到lerobot的对应features中
多层索引使用`.`分割
多数据结合用List[str]表示
例如:
map = {
    "observation.images.cam_high": "observation.cam_head.color",
    "observation.images.cam_left_wrist": "observation.cam_left_wrist.color",
    "observation.images.cam_right_wrist": "observation.cam_right_wrist.color",
    "observation.state": ["observation.left_arm.joint","observation.left_arm.gripper","observation.right_arm.joint","observation.right_arm.gripper"],
}
'''

class MyLerobotDataset:
    def __init__(self, repo_id: str, robot_type: str, fps: int, features: dict, map: dict, intruction_path: str):
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,   
        )
        self.map = map
        self.intruction_path = intruction_path

    def get_random_intruction(self, path=None):
        if path is None:
            with open(self.intruction_path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict['instructions']
                instruction = np.random.choice(instructions)
                return instruction
        else:
            with open(path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict['instructions']
                instruction = np.random.choice(instructions)
                return instruction

    def write(self, data: Dict,path=None):
        base_frame = {}
        if self.intruction_path is None:
            # 多任务, 需要匹配对应任务的指令
            instruction = self.get_random_intruction(path) 
        else:
            # 单任务, 读取默认指令位置
            instruction = self.get_random_intruction() 
        for key, value in self.map.items():
            base_frame[key] = np.array(get_item(data, value))
        episode_length = base_frame[list(base_frame.keys())[0]].shape[0]
        for i in range(episode_length):
            frame = {}
            for key, value in base_frame.items():
                frame[key] = value[i]   
            # 这个是最新版lerobot数据集格式才可以的, openpi版本的lerobot不支持strshuju,智能在save episode中写入
            frame["task"] = instruction  
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        # self.dataset.save_episode(task=instruction)
    
    # 新版lerobot没有该函数
    def consolidate(self):
        self.dataset.consolidate()
