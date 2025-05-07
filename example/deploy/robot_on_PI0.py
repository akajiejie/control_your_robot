import sys
sys.path.append("../../")

from my_robot.realman_dual_3_camera import MyRobot
from data.data import Data

import time
import keyboard

from policy.openpi.inference_model import PI0_DUAL

def transform_data(data):
    state = np.aray([data["left_arm"]["joint"], data["left_arm"]["gripper"], data["right_arm"]["joint"], data["right_arm"]["gripper"]])
    img_arr = data["cam_head"]["color"], data["cam_right_wrist"]["color"], data["cam_left_wrist"]["color"]
    return img_arr, state

if __name__ == "__main__":
    robot = MyRobot()
    robot.set_up()
    # load model
    # 双臂采用PI_DUAL, 单臂采用PI_SINGLE
    model = PI0_DUAL("model_path", "task_name")
    max_step = 1000
    num_episode = 10

    for i in range(num_episode):
        step = 0
        # 重置所有信息
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()

        # 等待允许执行推理指令, 按enter开始
        is_start = False
        while not is_start:
            if keyboard.is_pressed("enter"):
                is_start = True
                print("start to inference...")
            else:
                print("waiting for start command...")
                time.sleep(1)

        # 开始逐条推理运行
        while step < max_step:
            img_arr, state = robot.get_observation()
            action_chunk = model.update_observation_window(img_arr, state)
            for action in action_chunk:
                robot.move(action)
                step += 1
                time.sleep(1/robot.condition["save_interval"])

        print("finish episode", i)