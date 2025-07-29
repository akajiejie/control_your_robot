import sys
sys.path.append("./")
from controller import *
import time
from typing import *

from multiprocessing import Event, Semaphore

from utils.data_handler import debug_print

def RobotWorker(robot_class, start_episode,
                time_lock: Semaphore, start_event: Event, finish_event: Event, process_name: str):
    '''
    对于实现的机器人类进行多进程数据采集, 可以对多个机器人进行.
    输入:
    robot_class: 机器人类, my_robot::robot_class
    start_episode: 数据采集的开始序号, 只影响保存数据的后缀组号, int
    time_lock: 初始化对于当前组件的时间同步锁, 该锁需要分配给time_scheduler用于控制时间, multiprocessing::Semaphore
    start_event: 同步开始事件, 所有的组件共用一个, multiprocessing::Event
    finish_event: 同步结束事件, 所有的组件共用一个, multiprocessing::Event
    process_name:你希望当前进程叫什么, 用于对应子进程info的输出, str
    '''
    robot = robot_class(start_episode=start_episode)
    robot.set_up()
    
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
                data = robot.get()
                robot.collect(data)
            except Exception as e:
                debug_print(process_name, f"Error: {e}", "ERROR")

            debug_print(process_name, "Data processed. Waiting for next time slot.", "DEBUG")

        debug_print(process_name, "Finish event triggered. Finalizing...","INFO")
        robot.finish()
        debug_print(process_name, "Writing success!","DEBUG")
        
    except KeyboardInterrupt:
        debug_print(process_name, "Worker terminated by user.")
    finally:
        debug_print(process_name, "Worker exiting.")
