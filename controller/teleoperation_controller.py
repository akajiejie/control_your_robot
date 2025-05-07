import sys
sys.path.append("./")

import numpy as np

from controller.controller import Controller
from typing import Dict, Any

class TeleoperationController(Controller):
    def __init__(self):
        super().__init__()
        self.name = "arm_controller"
        self.controller = None
        self.controller_type = "teleoperator"
    
    def get_information(self):
        arm_info = {}
        state = self.get_state()
        if "end_pose" in self.collect_info:
            arm_info["end_pose"] = state["end_pose"]
        if "velocity" in self.collect_info:
            arm_info["velocity"] = state["velocity"]
        if "gripper" in self.collect_info:
            arm_info["gripper"] = state["gripper"]
        return arm_info
    
    def move(self, move_data:Dict[str, Any], is_delta=False):
        raise RuntimeError("teleoperator can not move by setting")

    def __repr__(self):
        if self.controller is not None:
            return f"{self.name}: \n \
                    controller: {self.controller}"
        else:
            return super().__repr__()