import pyrealsense2 as rs

def find_connected_realsense_devices():
    # 创建上下文对象
    ctx = rs.context()
    
    # 获取所有连接的设备
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("未检测到任何 RealSense 设备")
        return
    
    print(f"检测到 {len(devices)} 台 RealSense 设备:")
    
    for i, dev in enumerate(devices):
        # 获取设备信息
        serial_number = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        physical_port = dev.get_info(rs.camera_info.physical_port)
        
        print(f"\n设备 {i + 1}:")
        print(f"  名称: {name}")
        print(f"  序列号: {serial_number}")
        print(f"  物理端口: {physical_port}")

if __name__ == "__main__":
    find_connected_realsense_devices()