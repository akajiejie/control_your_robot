import sys
sys.path.append("./")

import os
import json
import time
import select
import numpy as np

from my_robot.agilex_piper_single import PiperSingle
from data.collect_any import CollectAny

from utils.time_scheduler import TimeScheduler
from utils.robot_worker import RobotWorker

ARM_INFO_NAME = ["qpos", "gripper"]

condition = {
    "save_path": "./datasets/ckpt",
    "task_name": "saving_move_path",
}

def is_enter_pressed():
    """非阻塞检测Enter键"""
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == '\n'

class PathCollector:
    def __init__(self, robot, condition, episode_index=0):
        # 传入的robot需要配置好对应需要采集的信息
        self.robot = robot
        self.collecter = CollectAny(condition, start_episode=0)
        self.condition = condition
        self.episode_index = episode_index  
    
    def collect(self):
        data = self.robot.get()
        self.collecter.collect(data[0], data[1])

    def save(self):
        json_data = {}

        # 遍历每个 episode，将 numpy 数组转换为列表
        for index, episode in enumerate(self.collecter.episode):
            episode_data = episode.copy()  # 复制数据，以避免直接修改原数据
            if isinstance(episode_data.get("left_arm", {}).get("qpos"), np.ndarray):
                # 将 ndarray 转换为列表
                episode_data["left_arm"]["qpos"] = episode_data["left_arm"]["qpos"].tolist()
            json_data[index] = episode_data
        
        print(json_data)
        
        save_path = os.path.join(self.condition["save_path"], f"{self.condition['task_name']}/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 保存到文件
        with open(os.path.join(save_path, f"{self.episode_index}.json"), "w") as f:
            json.dump(json_data, f, indent=4)
        self.collecter.episode = []
        self.episode_index += 1

    def play(self, robot, episode_index, is_block=False):
        path = os.path.join(self.condition['save_path'], f"{self.condition['task_name']}/{episode_index}.json")
        try:
            with open(path, "r") as f:
                json_data = json.load(f)
        except:
            print(f"{path} does not exist!")
            return
        
        i = 0 
        for episode in json_data.values():
            # 对应controller根据采集数据进行行动
            print(f"move {i}: {episode}")
            i += 1
            robot.move(episode)
            # 如果是非阻塞,当停止运动后进行下一步动作
            if not is_block:
                time.sleep(2)
            # 如果是阻塞的，可以添加一个等待时间，或阻塞完成直接继续
            if is_block:
                continue
        print("play finished!")
            
if __name__ == "__main__":
    robot = PiperSingle()
    robot.set_up()
    # 只收集对应的坐标和夹爪状态
    ARM_INFO_NAME = ["qpos", "gripper"]
    robot.set_collect_type(ARM_INFO_NAME, None) 
    collector = PathCollector(robot, condition, episode_index=0)
    '''
    按Enter键进行采集
    按Space键保存并退出
    保存的json文件可以删除不想要的ckpt,不会影响操作
    '''
    while True:
        user_input = input("请输入 'c' 收集数据，'s' 保存数据，'q' 退出采集: ").strip().lower()  # 获取用户输入
        if user_input == 'c':
            collector.collect()  # 执行收集数据操作
        elif user_input == 's':
            collector.save()  # 执行保存数据操作
            print("Collect finished!")
        elif user_input == 'q':
            print("Exiting...")
            break  # 退出循环
        else:
            print("无效输入，请重新输入！")

    # 测试运行，执行第0条轨迹
    collector.play(robot, 0, is_block=False)

    '''
    # 如果你机械臂只能在单个脚本中建立通讯,那么在本脚本中你可以添加一组数据采集器,注释上面的collector.play()
    
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
    
    collector.play(robot, 0, is_block=False)

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
    '''