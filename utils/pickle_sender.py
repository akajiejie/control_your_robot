import socket
import pickle

class Sender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)  # 确保端口是整数
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            print(f"[Sender] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[Sender] Connection failed: {e}")
            raise

    def send(self, data):
        try:
            serialized = pickle.dumps(data)
            length_prefix = len(serialized).to_bytes(4, 'big')
            self.socket.sendall(length_prefix + serialized)
        except Exception as e:
            print(f"[Sender] Send failed: {e}")
            raise

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self.socket.close()
        print("[Sender] Socket closed.")

if __name__ == "__main__":
    import numpy as np
    import time

    sender = Sender("127.0.0.1", "10000")
    try:
        for i in range(100):
            data = {
                "joint": np.random.rand(6) * 3.1415926,
                "gripper": 0.2
            }
            sender.send(data)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Sender] Interrupted by user.")
    finally:
        sender.close()
