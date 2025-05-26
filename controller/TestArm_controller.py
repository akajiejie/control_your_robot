import sys
sys.path.append("./")

from controller.arm_controller import ArmController

import numpy as np
import time

from utils.data_handler import debug_print

class TestArmController(ArmController):
    def __init__(self, name, DoFs=6,INFO="DEBUG"):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None
        self.INFO = INFO
        self.DoFs = DoFs
    
    def set_up(self):
        debug_print(f"{self.name}: setup success",self.INFO)

    def reset(self, start_state):
        if start_state.shape[0] == self.DoFs:
            debug_print(f"{self.name}: reset success, to start state \n {start_state}",self.INFO)
        else:
            debug_print(f"{self.name}: reset() input should be joint controll which dim is {self.DoFs}","ERROR")


    # 返回单位为米
    def get_state(self):
        state = {}
        
        # 获取对应信息
        state["joint"] = np.random.rand(self.DoFs) * 3.1515926
        state["qpos"] = np.random.rand(6)
        state["gripper"] = np.random.rand(1)
        debug_print(f"get state to \n {state}", self.INFO)
        return state

    # 单位为米
    def set_position(self, position):
        if position.shape[0] == 6:
            debug_print(f"{self.name}: using EULER set position to \n {position}", self.INFO)
        elif position.shape[0] == 7:
            debug_print(f"{self.name}: using QUATERNION set position to \n {position}", self.INFO)
        else:
            debug_print(f"{self.name}: set_position input size should be 6 -> EULER or 7 -> QUATERNION","ERROR")
    
    def set_joint(self, joint):
        if joint.shape[0] != self.DoFs:
            debug_print(f"{self.name}: set_joint() input size should be {self.DoFs}","ERROR")   
        else: 
            debug_print(f"{self.name}: set joint to \n {joint}", self.INFO)

    # 输入的是0~1的张合度
    def set_gripper(self, gripper):
        if isinstance(gripper, (int, float, complex)) and not isinstance(gripper, bool):
            if 1>gripper > 0:
                debug_print(f"{self.name}: set gripper to {gripper}", self.INFO)
            else:
                debug_print(f"{self.name}: gripper should be 0~1, but get {gripper}",self.INFO)
        else:
            debug_print(f"{self.name}: gripper should be a number 0~1","ERROR")
    
    def __del__(self):
        try:
            if hasattr(self, 'controller'):
                # Add any necessary cleanup for the arm controller
                pass
        except:
            pass

if __name__=="__main__":
    controller = TestArmController("test_arm",DoFs=6,INFO="DEBUG")

    controller.set_collect_info(["joint","qpos","gripper"])

    controller.set_up()

    controller.get_state()

    controller.set_gripper(0.2)

    controller.set_joint(np.array([0.1,0.1,-0.2,0.3,-0.2,0.5]))
    time.sleep(0.1)

    controller.set_position(np.array([0.057, 0.0, 0.260, 0.0, 0.085, 0.0]))
    time.sleep(0.1)

    controller.reset(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    controller.get()

    move_data = {
        "joint":np.random.rand(6) * 3.1515926,
        "gripper":0.2
    }
    controller.move(move_data)