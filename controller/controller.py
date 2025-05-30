import sys
sys.path.append("./")

from typing import List
import numpy as np

from utils.data_handler import debug_print

class Controller:
    def __init__(self):
        self.name = "controller"
        self.controller_type = "base_controller"
        self.is_set_up = False
    
    def set_collect_info(self, collect_info:List[str]):
        self.collect_info = collect_info

    # get controller infomation
    def get(self):
        if self.collect_info is None:
            raise ValueError("collect_info is not set")
        info = self.get_information()
        for collect_info in self.collect_info:
            if info[collect_info] is None:
                debug_print(f"{self.name}", f"gripper information is None", "ERROR")

        return {collect_info: info[collect_info] for collect_info in self.collect_info}

    def move(self, move_data, is_delta=False):
        # if not set(move_data.keys()).issubset(self.collect_info):
        #     raise ValueError(f"Invalid keys in move_data. Valid keys are: {self.collect_info}")
        try:
            self.move_controller(move_data, is_delta)
        except Exception as e:
            print(f"move error: {e}")
    
   # init controller
    def set_up(self):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    # print controller
    def __repr__(self):
        return f"Base Controller, can't be used directly \n \
                name: {self.name} \n \
                controller_type: {self.controller_type}"
