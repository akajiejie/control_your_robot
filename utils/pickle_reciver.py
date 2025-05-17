import socket
import pickle
from threading import Thread

class Receiver:
    def __init__(self, host: str, port: int, handler):
        self.host = host
        self.port = port
        self.handler = handler
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(1)

    def start(self):
        print(f"[Receiver] Listening on {self.host}:{self.port}")
        conn, addr = self.server.accept()
        print(f"[Receiver] Connected by {addr}")
        Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            while True:
                length_bytes = conn.recv(4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet:
                        break
                    data += packet
                message = pickle.loads(data)
                self.handler(message)
        finally:
            conn.close()
