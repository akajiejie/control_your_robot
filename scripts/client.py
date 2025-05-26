import sys
sys.path.append('./')

from utils.pickle_sender import Sender
from utils.pickle_reciever import Reciever

import socket
import time
import numpy as np

def transform_data(data):
    state = np.aray([data["left_arm"]["joint"], data["left_arm"]["gripper"], data["right_arm"]["joint"], data["right_arm"]["gripper"]])
    img_arr = data["cam_head"]["color"], data["cam_right_wrist"]["color"], data["cam_left_wrist"]["color"]
    return img_arr, state

class client:
    def __init__(self,robot,cntrol_freq=10):
        self.robot = robot
        self.cntrol_freq = cntrol_freq
    
    def set_up(self, server_ip, server_port, recievee_ip, reciever_port):
        self.sender = Sender(server_ip, server_port)
        self.reciever = Reciever(recievee_ip, reciever_port,self.robot.move())       

        self.receiver.start()

def move(self, message):
    action_chunk = message["action_chunk"]
    action_chunk = np.array(action_chunk)

    for action in action_chunk:
        left_action = action[:7]
        right_action = action[-7:]
        move_data = {
            "left_arm":left_action,
            "right_arm":right_action,
        }
        self.robot.move()

    def play_once(self):
        raw_data = self.robot.get()
        img_arr, state = transform_data(raw_data)
        data_send = {
            "img_arr",img_arr,
            "state",state
        }
        # 发送数据
        self.sender.send(data_send)
        time.sleep(1 / self.cntrol_freq)
    
    def close(self):
        self.sender.close()
        self.receiver.close()


if __name__ == "__main__":
    pass