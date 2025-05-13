import socket
import pickle
import cv2

def get_robot_state():
    # 模拟函数，实际可替换成真实获取状态的代码
    return {"joint_angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

def send_data_and_receive_action(server_ip, port):
    # 建立 socket 连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, port))

    # 获取图像和状态
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    robot_state = get_robot_state()

    # 打包数据
    data = pickle.dumps({
        "image": frame,
        "state": robot_state
    })

    # 发送数据长度和数据
    client.sendall(len(data).to_bytes(8, byteorder='big'))
    client.sendall(data)

    # 接收返回动作
    action_len = int.from_bytes(client.recv(8), byteorder='big')
    action_data = b''
    while len(action_data) < action_len:
        action_data += client.recv(action_len - len(action_data))

    action = pickle.loads(action_data)
    print("Received action:", action)

    client.close()
