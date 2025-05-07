import sys
sys.path.append("./")

import select
from my_robot.agilex_piper_single import PiperSingle
import time

def is_enter_pressed():
    """非阻塞检测Enter键"""
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == '\n'

if __name__ == "__main__":
    robot = PiperSingle()
    robot.set_up()
    num_episode = 10
    robot.condition["task_name"] = "my_test"

    for _ in range(num_episode):
        robot.reset()
        print("等待开始...")
        while not robot.is_start():
            time.sleep(1/robot.condition["save_interval"])
        
        print("开始采集,按Enter停止...")
        while True:
            data = robot.get()
            robot.collect(data)
            
            if is_enter_pressed():
                robot.finish()
                break
                
            time.sleep(1/robot.condition["save_interval"])