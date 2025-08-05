import sys
sys.path.append("./")

import time
from multiprocessing import Manager, Event

from utils.data_handler import is_enter_pressed
from utils.time_scheduler import TimeScheduler
from utils.worker import Worker
from controller.TestArm_controller import TestArmController

from my_robot.test_robot import TestRobot 

class MasterWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event):
        super().__init__(process_name, start_event, end_event)
        self.manager = Manager()
        self.data_buffer = self.manager.dict()
    
    def handler(self):
        data = self.component.get()
        for key, value in data.items():
            self.data_buffer[key] = value

    def component_init(self):
        self.component = TestArmController("left_teleop_arm")
        self.component.set_up()
        self.component.set_collect_info(["joint", "gripper"])

class SlaveWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, move_data_buffer: Manager):
        super().__init__(process_name, start_event, end_event)
        self.move_data_buffer = move_data_buffer
    
    def handler(self):
        move_data = dict(self.move_data_buffer)
        self.component.move({"arm": 
                                {
                                    "left_arm": move_data
                                }
                            })

        data = self.component.get()
        self.component.collect(data)
    
    def component_init(self):
        self.component = TestRobot()
        self.component.set_up()
    
    def finish(self):
        self.component.finish()

if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"
    num_episode = 3
    avg_collect_time = 0

    start_episode = 0

    for i in range(num_episode):
        is_start = False

        start_event, end_event = Event(), Event()

        master = MasterWorker("master_arm", start_event, end_event)
        slave = SlaveWorker("slave_arm", start_event, end_event, master.data_buffer)
        
        time_scheduler = TimeScheduler([master.forward_event], time_freq=10, end_Events=[slave.next_event])
        
        master.next_to(slave)

        master.start()
        slave.start()

        while not is_start:
            time.sleep(0.01)
            if is_enter_pressed():
                is_start = True
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