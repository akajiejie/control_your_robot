import socket
import pickle

class Sender:
    def __init__(self, send_socket):
        self.socket = send_socket

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
    host = "127.0.0.1"
    port = 10000
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect((host, port))
    sender = Sender(send_socket)
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
