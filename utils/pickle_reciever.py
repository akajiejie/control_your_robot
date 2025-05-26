import socket
import pickle
from threading import Thread

class Reciever:
    def __init__(self, host: str, port: int, handler):
        self.host = host
        self.port = int(port)
        self.handler = handler
        self.conn_closed = False
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许重用地址，避免TIME_WAIT影响绑定
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self._running = True
        self._conn_thread = None

    def start(self):
        print(f"[Receiver] Listening on {self.host}:{self.port}")
        try:
            conn, addr = self.server.accept()
            print(f"[Receiver] Connected by {addr}")
            self._conn_thread = Thread(target=self._handle, args=(conn,), daemon=True)
            self._conn_thread.start()
        except Exception as e:
            print(f"[Receiver] Exception on accept: {e}")
            self.conn_closed = True

    def _handle(self, conn):
        try:
            while True:
                length_bytes = conn.recv(4)
                if not length_bytes:
                    print("[Receiver] Connection closed by client.")
                    self.conn_closed = True
                    break
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet:
                        print("[Receiver] Connection closed during data recv.")
                        self.conn_closed = True
                        break
                    data += packet
                if self.conn_closed:
                    break
                message = pickle.loads(data)
                self.handler(message)
        except Exception as e:
            print(f"[Receiver] Exception in handler: {e}")
            self.conn_closed = True
        finally:
            conn.close()
            self.conn_closed = True

    def close(self):
        self._running = False
        try:
            self.server.close()
        except Exception:
            pass
        print("[Receiver] Server socket closed.")

def handler(message):
    print("[Handler] Received:", message)

if __name__ == "__main__":
    import time

    receiver = Reciever("127.0.0.1", 10000, handler=handler)
    receiver.start()

    while True:
        if receiver.conn_closed:
            print("[Main] Connection closed, exiting program.")
            break
        time.sleep(1)
