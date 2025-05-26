import sys
sys.path.append('./')

from policy.test_policy.inference_model import TestModel
from utils.pickle_sender import Sender
from utils.pickle_reciever import Reciever

import socket
import time


class Server:
    def __init__(self,model,cntrol_freq=10):
        self.cntrol_freq = cntrol_freq
        self.model = model
    
    def set_up(self, send_ip, send_port, recieve_ip, reciever_port):
        self.send_ip = send_ip
        self.send_port = send_port
        self.sender = Sender(self.send_ip, self.send_port)
        self.reciever = Reciever(recieve_ip, reciever_port,self.infer)
        self.reciever.start()
    
    def infer(self, message):
        if not hasattr(self, 'sender'):
            self.sender = Sender(self.send_ip, self.send_port)
        print(message)
        img_arr, state = message["img_arr"], message["state"]
        self.model.update_observation_window(img_arr, state)
        action_chunk = self.model.get_action()
        self.sender.send({"action_chunk":action_chunk})
    
    def close(self):
        self.sender.close()

if __name__ == "__main__":
    DoFs = 6
    model = TestModel("test",DoFs=(2*(DoFs + 1)))

    server = Server(model)
    server.set_up("127.0.0.1","10001","127.0.0.1","10000")

    while True:
        if server.reciever.conn_closed:
            print("[Main] Connection closed, exiting program.")
            break
        time.sleep(1)
