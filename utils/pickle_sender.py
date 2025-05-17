import socket
import pickle

class Sender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
    
    def send(self, data):
        serialized = pickle.dumps(data)
        self.socket.sendall(len(serialized).to_bytes(4, 'big') + serialized)

    def close(self):
        self.socket.close()
