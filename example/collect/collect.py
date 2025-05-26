import sys
sys.path.append("./")

import select

from my_robot.test_robot import TestRobot

import time

from utils.data_handler import is_enter_pressed


if __name__ == "__main__":
    robot = TestRobot()
    robot.set_up()
    num_episode = 10
    robot.condition["task_name"] = "my_test"

    for _ in range(num_episode):
        robot.reset()
        print("等待开始...")
        while not robot.is_start() or not is_enter_pressed():
            time.sleep(1/robot.condition["save_interval"])
        
        print("开始采集,按Enter停止...")
        while True:
            data = robot.get()
            robot.collect(data)
            
            if is_enter_pressed():
                robot.finish()
                break
                
            time.sleep(1/robot.condition["save_interval"])