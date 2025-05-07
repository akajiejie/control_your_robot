import sys
sys.path.append("./")
from controller import *
import time
from typing import *

from multiprocessing import Event, Semaphore

DEBUG = False
Release = True

def debug_print(process_name, msg, Release=False):
    if DEBUG or Release:
        print(f"[{process_name}] {msg}")

def RobotWorker(robot_class, 
                time_lock: Semaphore, start_event: Event, finish_event: Event, process_name: str):
    """
    工作进程函数：
    - 通过 Semaphore 控制时间同步
    - 通过 Event 通知进程终止
    """
    robot = robot_class()
    robot.set_up()
    while not start_event.is_set():
        debug_print(process_name ,"Press Enter to start...",Release)
        time.sleep(1)
    
    debug_print(process_name, "Get start Event, start collecting...",Release)
    debug_print(process_name, "To finish this episode, please press Enter. ",Release)
    try:
        while not finish_event.is_set():
            time_lock.acquire()  # 阻塞等待调度令牌
            if finish_event.is_set():
                break  # 防止在 acquire 之后立即退出之前处理数据

            debug_print(process_name, "Time lock acquired. Processing data...")

            try:
                data = robot.get()
                robot.collect(data)
            except Exception as e:
                debug_print(process_name, f"Error: {e}", Release)

            debug_print(process_name, "Data processed. Waiting for next time slot.")

        # 安全终止处理
        debug_print(process_name, "Finish event triggered. Finalizing...",Release)
        robot.finish()
        debug_print(process_name, "Writing success!",Release)
        
    except KeyboardInterrupt:
        debug_print(process_name, "Worker terminated by user.")
    finally:
        debug_print(process_name, "Worker exiting.")
