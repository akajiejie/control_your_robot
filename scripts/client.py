import sys
sys.path.append('./')

from my_robot.test_robot import TestRobot
from utils.pickle_sender import Sender
from utils.pickle_reciever import Reciever

import socket
import time
import numpy as np

def input_transform(data):
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    ])


    img_arr = data[1]["cam_head"]["color"], data[1]["cam_right_wrist"]["color"], data[1]["cam_left_wrist"]["color"]
    return img_arr, state

def output_transform(data):
    move_data = {
        "left_arm":{
            "joint":data[:6],
            "gripper":data[6]
        },
        "right_arm":{
            "joint":data[7:13],
            "gripper":data[13]
        }
    }
    return move_data

class Client:
    def __init__(self,robot,cntrol_freq=10):
        self.robot = robot
        self.cntrol_freq = cntrol_freq
    
    def set_up(self, send_ip, send_port, recieve_ip, reciever_port):
        self.send_ip = send_ip
        self.send_port = send_port

        # self.sender = Sender(self.send_ip, self.send_port)

        self.receiver = Reciever(recieve_ip, reciever_port,self.move)
        self.receiver.start()

    def move(self, message):
        action_chunk = message["action_chunk"]
        action_chunk = np.array(action_chunk)

        for action in action_chunk:
            move_data = output_transform(action)
            self.robot.move(move_data)

    def play_once(self):
        if not hasattr(self, 'sender'):
            self.sender = Sender(self.send_ip, self.send_port)
        
        raw_data = self.robot.get()
        img_arr, state = input_transform(raw_data)
        data_send = {
            "img_arr": img_arr,
            "state": state
        }

        # 发送数据
        self.sender.send(data_send)
        time.sleep(1 / self.cntrol_freq)

    def close(self):
        if hasattr(self,'sender'):
            self.sender.close()

if __name__ == "__main__":
    DoFs = 6
    robot = TestRobot(DoFs=DoFs, INFO="DEBUG")
    robot.set_up()

    client = Client(robot)
    client.set_up("127.0.0.1","10000","127.0.0.1","10001")
    for i in range(10):
        try:
            print("play once")
            client.play_once()
            time.sleep(1)
        except:
            client.close()
    client.close()