import time
import multiprocessing as mp
from multiprocessing import Array, Process, Semaphore, Lock, Value
from typing import List
import numpy as np

DEBUG = False
Release = True

from utils.data_handler import debug_print

def worker(process_id: int, process_name: str, time_semaphore: Semaphore, result_array: Array, result_lock: Lock):
    while True:
        # Block until the scheduler issues an "execution token"
        time_semaphore.acquire()  
        debug_print(process_name, "received time slot", "DEBUG")

        # Simulate task processing + data writing (with locking)
        result = np.random.randn(5,5).flatten()
        with result_lock:
            result_array[process_id*25:(process_id+1)*25] = result
        time.sleep(0.01) 

class TimeScheduler:
    def __init__(self, time_semaphores: List[Semaphore], time_freq=10):
        self.time_freq = time_freq
        self.time_semaphores = time_semaphores
        self.process_name = "time_scheduler"
        self.real_time_accumulate_time_interval = Value('d', 0.0)
        self.step = Value('i', 0)

    def time_worker(self):
        last_time = time.monotonic()
        while True:
            now = time.monotonic()
            if now - last_time >= 1 / self.time_freq:
                if all(sem.get_value() == 0 for sem in self.time_semaphores):
                    for sem in self.time_semaphores:
                        sem.release()  
                        debug_print(self.process_name, "released time slot to one worker", "DEBUG")
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
        self.time_locker = Process(target=self.time_worker)
        self.time_locker.start()

    def stop(self):
        for sem in self.time_semaphores:
                sem.release()  # 防止卡在获取锁
        self.time_locker.terminate()
        self.time_locker.join()
        self.time_locker.close()
        self.time_locker = None
        with self.real_time_accumulate_time_interval.get_lock():
            self.real_time_average_time_interval = self.real_time_accumulate_time_interval.value / self.step.value
        debug_print(self.process_name, f"average real time collect interval is: {self.real_time_accumulate_time_interval.value / self.step.value}", "INFO")

if __name__ == "__main__":
    processes = []
    process_num = 4
    result_array = Array('d', process_num * 25)
    time_semaphores = [Semaphore(0) for _ in range(process_num)]
    result_lock = Lock()

    # start workers
    for i in range(process_num):
        process = Process(target=worker, args=(i, f"process_{i}", time_semaphores[i], result_array, result_lock))
        process.start()
        processes.append(process)

    # start time scheduler
    time_scheduler = TimeScheduler(time_semaphores, time_freq=10)
    time_scheduler.start()

    # (optional)
    try:
        time.sleep(5)
        print("Sample result snapshot:", result_array[:])
    except KeyboardInterrupt:
        print("Main process interrupted.")
    finally:
        time_scheduler.stop()
        for p in processes:
            p.terminate()
            p.join()
