import sys
sys.path.append("./")
from controller import *
import time
from typing import *

from multiprocessing import Event, Semaphore, Process, Value, Manager

from utils.data_handler import debug_print

class DataBuffer:
    def __init__(self, manager):
        # 用 Manager dict 而不是普通 dict
        self.manager = manager
        self.buffer = manager.dict()

    def collect(self, name, data):
        if name not in self.buffer:
            self.buffer[name] = self.manager.list()
        self.buffer[name].append(data)

    def get(self):
        return dict(self.buffer)
        
def ComponentWorker(component_class, component_name, component_setup_input,component_collect_info, data_buffer: DataBuffer,
                time_lock: Semaphore, start_event: Event, finish_event: Event, process_name: str):
    """
    Worker process function:
    - Uses a Semaphore to control time synchronization
    - Uses an Event to signal process termination
    """
    component = component_class(component_name)
    
    if not component_setup_input is None:
        component.set_up(*component_setup_input)
    else:
        component.set_up()
    
    component.set_collect_info(component_collect_info)
    
    last_time = time.monotonic()
    while not start_event.is_set():
        now = time.monotonic()
        if now - last_time > 5:  
            debug_print(process_name ,"Press Enter to start...","INFO")
            last_time = now
        
    debug_print(process_name, "Get start Event, start collecting...","INFO")
    debug_print(process_name, "To finish this episode, please press Enter. ","INFO")
    try:
        while not finish_event.is_set():
            time_lock.acquire()  
            if finish_event.is_set():
                break  # Prevent exiting immediately after acquire before processing data

            debug_print(process_name, "Time lock acquired. Processing data...", "DEBUG")

            try:
                data = component.get()
                data_buffer.collect(component.name, data)
            except Exception as e:
                debug_print(process_name, f"Error: {e}", "ERROR")

            debug_print(process_name, "Data processed. Waiting for next time slot.", "DEBUG")

        debug_print(process_name, "Finish event triggered. Finalizing...","INFO")
        
    except KeyboardInterrupt:
        debug_print(process_name, "Worker terminated by user.")
    finally:
        debug_print(process_name, "Worker exiting.")
    
if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"

    from sensor.TestVision_sensor import TestVisonSensor
    from controller.TestArm_controller import TestArmController
    from utils.time_scheduler import TimeScheduler
    from utils.data_handler import is_enter_pressed

    # 初始化共享操作
    processes = []
    start_event = Event()
    finish_event = Event()
    manager = Manager()
    data_buffer = DataBuffer(manager)

    time_lock_vision = Semaphore(0)
    time_lock_arm = Semaphore(0)
    vision_process = Process(target=ComponentWorker, args=(TestVisonSensor, "test_vision", None, ["color"], data_buffer, time_lock_vision, start_event, finish_event, "vision_worker"))
    arm_process = Process(target=ComponentWorker, args=(TestArmController, "test_arm", None, ["joint", "qpos", "gripper"], data_buffer, time_lock_arm, start_event, finish_event, "arm_worker"))
    time_scheduler = TimeScheduler([time_lock_vision, time_lock_arm], time_freq=100) # 可以给多个进程同时上锁
    
    processes.append(vision_process)
    processes.append(arm_process)

    for process in processes:
        process.start()

    is_start = False

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
            finish_event.set()  
            time_scheduler.stop()  
            is_start = False
    
    # 销毁多进程
    for process in processes:
        if process.is_alive():
            process.join()
            process.close()
    