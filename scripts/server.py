import sys
sys.path.append('./')

from utils.pickle_sender import Sender
from utils.pickle_reciever import Reciever

import socket
import time

class server:
    def __init__(self,model,cntrol_freq=10):
        self.cntrol_freq = cntrol_freq
        self.model = model
    
    def set_up(self, server_ip, server_port, recievee_ip, reciever_port):
        self.sender = Sender(server_ip, server_port)
        self.reciever = Reciever(recievee_ip, reciever_port,self.infer())       

        self.receiver.start()
    
    def infer(self, message):
        img_arr, state = message["img_arr"], message["state"]
        self.model.update_observation_windows(img_arr, state)
        action_chunk = self.model.get_action()
        self.sender.send({"action_chunk":action_chunk})
    
    def close(self):
        self.sender.close()
        self.receiver.close()