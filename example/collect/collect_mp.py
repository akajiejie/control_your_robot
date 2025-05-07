import sys
sys.path.append("./")
import time
import select

from multiprocessing import Array, Process, Lock, Event, Semaphore

from utils.time_scheduler import TimeScheduler
from utils.robot_worker import RobotWorker
from my_robot.agilex_piper_single import PiperSingle

def is_enter_pressed():
    """非阻塞检测Enter键"""
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == '\n'

if __name__ == "__main__":
    num_episode = 10

    for i in range(num_episode):
        is_start = False
        
        # 重置进程
        time_lock = Semaphore(0)
        start_event = Event()
        finish_event = Event()
        robot_process = Process(target=RobotWorker, args=(PiperSingle, time_lock, start_event, finish_event, "robot_worker"))
        time_scheduler = TimeScheduler([time_lock], time_interval=10) # 可以给多个进程同时上锁
        
        robot_process.start()
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
        if robot_process.is_alive():
            robot_process.join()
            robot_process.close()


        

        
