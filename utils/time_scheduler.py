import time
import multiprocessing as mp
from multiprocessing import Array, Process, Lock, Value, Event, Barrier
from typing import List
import numpy as np

DEBUG = False
Release = True

from utils.data_handler import debug_print

class TimeScheduler:
    '''
    时间控制器, 用于同步不同进程之间的信号量
    work_barrier: 每个子进程的控制都需要有一个信号量控制循环操作, 这里就是将所有进程的信号量进行控制, List[Event]
    time_freq: 采集数据的频率, 实际频率可能会稍微低于该频率, 会保存最终采集平均时间间隔在数据采集的config里, int
    '''
    def __init__(self, work_barrier: Barrier, time_freq=10, end_events=None):
        self.time_freq = time_freq
        self.work_barrier = work_barrier
        self.end_events = end_events
        self.process_name = "time_scheduler"
        self.real_time_accumulate_time_interval = Value('d', 0.0)
        self.step = Value('i', 0)

    def time_worker(self):
        last_time = time.monotonic()

        i = 0
        while True:
            now = time.monotonic()
            if now - last_time >= 1 / self.time_freq:
                    
                if  self.end_events is None:
                    try:
                        self.work_barrier.wait()
                    except Exception as e:
                        debug_print(self.process_name, f"{e}", "WARNING")
                        return
                else:
                    while True:
                        if all(event.is_set() for event in self.end_events):
                            break
                    
                debug_print(self.process_name, f"the actual time interval is {now - last_time}", "DEBUG")
                with self.real_time_accumulate_time_interval.get_lock():
                    self.real_time_accumulate_time_interval.value = self.real_time_accumulate_time_interval.value + (now - last_time)
                with self.step.get_lock():
                    self.step.value += 1

                if now - last_time > 2 / self.time_freq:
                        debug_print(self.process_name, "The current lock release time has exceeded twice the intended time interval.\n Please check whether the corresponding component's get() function is taking too long.", "WARNING")
                        debug_print(self.process_name, f"the actual time interval is {now - last_time}", "WARNING")
                last_time = now

    def start(self):
        '''
        开启时间同步器进程
        '''
        self.time_locker = Process(target=self.time_worker)
        self.time_locker.start()
        self.work_barrier.wait()

    def stop(self):
        '''
        释放该时间同步器进程
        '''
        self.time_locker = None
        with self.real_time_accumulate_time_interval.get_lock():
            self.real_time_average_time_interval = self.real_time_accumulate_time_interval.value / self.step.value
        debug_print(self.process_name, f"average real time collect interval is: {self.real_time_accumulate_time_interval.value / self.step.value}", "INFO")

if __name__ == "__main__":
    pass