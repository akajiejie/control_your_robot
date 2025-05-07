import sys
import termios
import tty
import select
import time

def is_enter_pressed():
    """非阻塞检测Enter键"""
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == '\n'

def is_space_pressed():
    """非阻塞检测空格键"""
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == ' '

# 示例主循环
while True:
    if is_enter_pressed():
        print("Enter pressed")
        # 处理Enter键逻辑
    elif is_space_pressed():
        print("Space pressed")
        # 处理空格键逻辑
        break
    else:
        time.sleep(1 / 10)  # 控制循环间隔，不占用过多 CPU
