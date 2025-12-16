from re import S, T
import sys

from h5py._hl.dataset import sel
sys.path.append("./")

import time
from multiprocessing import Manager, Event

from utils.data_handler import is_enter_pressed
from utils.time_scheduler import TimeScheduler
from utils.worker import Worker
from data.collect_any import CollectAny
from controller.drAloha_controller import DrAlohaController
from my_robot.agilex_piper_dual_base import PiperDual
import math
from typing import Dict, Any

condition = {
    "save_path": "./save/", 
    "task_name": "test_dual", 
    "save_format": "hdf5", 
    "save_freq": 30,
    "collect_type": "teleop",
}


class MasterWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event):
        super().__init__(process_name, start_event, end_event)
        self.manager = Manager()
        self.data_buffer = self.manager.dict()
        self.gravity_update_interval = 0.1  # 10Hz 更新频率
        self.last_gravity_update = 0
        self.start_gravity = False
        self.zero_gravity_flag = self.manager.Value('b', False)
        self.gravity_update_error = self.manager.Value('b', False)  # 重力补偿更新错误标志
    def handler(self):
        # 检查是否需要执行零重力（只在启动时调用一次）
        if self.zero_gravity_flag.value:
            self.component_left.zero_gravity()
            self.component_right.zero_gravity()
            self.zero_gravity_flag.value = False
            self.start_gravity = True  # 零重力调用后开启重力补偿
            
        left_arm_data = self.component_left.get()
        right_arm_data = self.component_right.get()
        
        # 分别对左右臂数据进行转换
        left_action = self.action_transform(left_arm_data)
        right_action = self.action_transform(right_arm_data)
        
        # 拼接成双臂数据字典
        data = {
            "left_arm": left_action,
            "right_arm": right_action
        }

        if self.start_gravity :
            current_time = time.time()
            if current_time - self.last_gravity_update >= self.gravity_update_interval:
                result_left = self.component_left.update_gravity()
                result_right = self.component_right.update_gravity()
                self.last_gravity_update = current_time
                
                # 检查update_gravity是否报错（返回None表示重试多次后失败）
                # 当result不为None且大于1时，说明重试了多次才成功
                if (result_left is not None and result_left > 1) or (result_right is not None and result_right > 1):
                    self.gravity_update_error.value = True
                else:
                    self.gravity_update_error.value = False
        else:
            pass
        
        
        for key, value in data.items():
            self.data_buffer[key] = value

    def component_init(self):
        self.component_left = DrAlohaController(name="left_arm")
        self.component_right = DrAlohaController(name="right_arm")
        self.component_left.set_up(com="/dev/ttyACM0")
        self.component_right.set_up(com="/dev/ttyACM1")
        self.component_left.set_collect_info(["joint","gripper"])
        self.component_right.set_collect_info(["joint","gripper"])
        self.component_left.apply_calibration()
        self.component_right.apply_calibration()
        
    def action_transform(self, move_data:Dict[str, Any]):
        """ Transform the action from master arm to the slave arm."""
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
        # 直接使用 NumPy 数组，无需转换
        joints = move_data["joint"].copy()
        
        joints[1] = joints[1] - math.radians(90)   # 关节2校准：减去90°
        joints[2] = joints[2] + math.radians(175)  # 关节3校准：增加175°
        
        for i in [1, 2, 4]:  # 关节2、3、5需反转方向
            joints[i] = -joints[i]
        left_joints = [
            clamp(joints[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
        ]
        left_joints = [float(joint) for joint in left_joints]
        # gripper_data = move_data["gripper"]
        gripper = move_data["gripper"]
        
        action = {
                "joint": left_joints,
                "gripper": gripper,
        }
        return action
    def finish(self):
        self.start_gravity = False
        self.zero_gravity_flag.value = False
        self.gravity_update_error.value = False
        for i in range(1,7):
            self.component_left.controller.estop(i)
            self.component_right.controller.estop(i)
        return super().finish()
class SlaveWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, move_data_buffer: Manager, gravity_update_error):
        super().__init__(process_name, start_event, end_event)
        self.move_data_buffer = move_data_buffer
        self.gravity_update_error = gravity_update_error  # 接收主臂的重力补偿错误标志
        self.manager = Manager()
        self.data_buffer = self.manager.dict()
    
    def handler(self):
        move_data = dict(self.move_data_buffer)
     
        # 当主臂重力补偿更新重试次数>1时，打印move_data用于调试
        if self.gravity_update_error.value:
            print("move_data:",move_data)
            
        if move_data is not None or self.gravity_update_error.value is not False:
            left_move_data = move_data["left_arm"]
            right_move_data = move_data["right_arm"]
            self.component.move({"arm": 
                                    {
                                        "left_arm": left_move_data,
                                        "right_arm": right_move_data,
                                    }
                                })
            self.gravity_update_error.value = False  # reset error flag

        data = self.component.get()

        self.data_buffer["controller"] = self.manager.dict()
        self.data_buffer["sensor"] = self.manager.dict()
        # self.data_buffer["controller"]["master_left_arm"] = self.manager.dict()

        for key, value in data[0].items():
            self.data_buffer["controller"]["slave_"+key] = value
        
        for key, value in data[1].items():
            self.data_buffer["sensor"]["slave_"+key] = value

        # for key, value in move_data.items():
        #     self.data_buffer["controller"]["master_left_arm"][key] = value
    
    def component_init(self):
        self.component = PiperDual()
        self.component.set_up()

        self.component.reset()  


class DataWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, collect_data_buffer: Manager, episode_id=0, resume=False):
        super().__init__(process_name, start_event, end_event)
        self.collect_data_buffer = collect_data_buffer
        self.episode_id = episode_id
        self.resume = resume
    def component_init(self):
        self.collection = CollectAny(condition=condition, start_episode=self.episode_id, move_check=True, resume=self.resume)
    
    def handler(self):
        data = dict(self.collect_data_buffer)
        self.collection.collect(data["controller"], data["sensor"])
    
    def finish(self):
        self.collection.write()

if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    num_episode = 1
    avg_collect_time = 0

    for i in range(num_episode):
        is_start = False

        start_event, end_event = Event(), Event()
        
        master = MasterWorker("master_arm", start_event, end_event)
        time.sleep(1)
        slave = SlaveWorker("slave_arm", start_event, end_event, master.data_buffer, master.gravity_update_error)
        data = DataWorker("collect_data", start_event, end_event, slave.data_buffer, episode_id=i, resume=True)

        time_scheduler = TimeScheduler(work_events=[master.forward_event], time_freq=30, end_events=[data.next_event])
        
        master.next_to(slave)
        slave.next_to(data)

        master.start()
        slave.start()
        data.start()

        while not is_start:
            time.sleep(0.01)
            if is_enter_pressed():
                is_start = True
                master.zero_gravity_flag.value = True
                start_event.set()
                
            else:
                time.sleep(1)

        time_scheduler.start()
        while is_start:
            time.sleep(0.01)
            if is_enter_pressed():
                end_event.set()  
                time_scheduler.stop()  
                is_start = False

        # 给数据写入一定时间缓冲
        time.sleep(2)

        master.stop()
        slave.stop()
        data.stop()