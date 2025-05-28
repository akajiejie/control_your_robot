import sys
sys.path.append("../../")

from my_robot.agilex_piper_single import PiperSingle

import time
import keyboard

from policy.RDT.inference_model import RDT

import numpy as np 


def transform_data(data):
    state = np.aray([data["left_arm"]["joint"], data["left_arm"]["gripper"], data["right_arm"]["joint"], data["right_arm"]["gripper"]])
    img_arr = data["cam_head"]["color"], data["cam_right_wrist"]["color"], data["cam_left_wrist"]["color"]
    return img_arr, state

if __name__ == "__main__":
    robot = PiperSingle()
    robot.set_up()
    # load model
    policy = RDT("model_path", "task_name")
    max_step = 1000
    num_episode = 10
    for i in range(num_episode):
        step = 0
        while True:
            if keyboard.is_pressed("enter"):
                print("reset robot")
                break
            else:
                print("waiting for command to reset robot")

        robot.reset()
        RDT.reset_obsrvationwindows()
        RDT.random_set_language()
        is_start = False
        while not is_start:
            if keyboard.is_pressed("enter"):
                is_start = True
                print("start to inference...")
        
        while step < max_step:
            # 更新robot的observation
            data = robot.get()
            img_arr, state = transform_data(data)
            policy.update_observation_window(img_arr, state)
            # 进行推理
            action_chunk = robot.get_action()
            #执行推理结果
            for action in action_chunk:
                robot.move(action)
                step += 1
                time.sleep(1/robot.condition["save_interval"])

        print(f"finish episode {i}")