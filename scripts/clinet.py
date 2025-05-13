import socket
import pickle
import numpy as np

def infer_action(image, state):
    # 返回示例
    return {"actions": [0,0,0,0,0,0;0,
                        0,0,0,0,0,0,0]}  # 示例动作

def run_server(host='', port=8888):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print("Server listening on port", port)

    while True:
        conn, addr = server.accept()
        print("Connection from", addr)

        # 接收数据长度
        data_len = int.from_bytes(conn.recv(8), byteorder='big')

        # 接收数据本体
        data = b''
        while len(data) < data_len:
            data += conn.recv(data_len - len(data))

        data = pickle.loads(data)
        image = data['image']
        state = data['state']

        action = infer_action(image, state)

        # 返回动作
        action_bytes = pickle.dumps(action)
        conn.sendall(len(action_bytes).to_bytes(8, byteorder='big'))
        conn.sendall(action_bytes)

        conn.close()
