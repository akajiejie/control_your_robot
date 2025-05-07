import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import time

def save_realsense_images():
    # 创建输出目录
    output_dir = "realsense_captures"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建管道和配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("419522072373")
    
    # 启用彩色和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动管道
    pipeline.start(config)
    
    # 等待第一帧
    # print("等待帧稳定...")
    # for _ in range(30):  # 跳过前30帧让自动曝光稳定
    #     pipeline.wait_for_frames()
    
    # 获取帧
    data = []
    for i in range(1000):
        frames = pipeline.wait_for_frames()
        print(i)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
    
        if not color_frame or not depth_frame:
            raise RuntimeError("无法获取帧数据")
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        data.append(color_image)
        time.sleep(0.1)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存彩色图像
    color_filename = os.path.join(output_dir, f"color_{timestamp}.png")
    cv2.imwrite(color_filename, color_image)
    print(f"彩色图像已保存: {color_filename}")
    
    # 保存深度图像（伪彩色可视化）
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), 
        cv2.COLORMAP_JET
    )
    depth_filename = os.path.join(output_dir, f"depth_{timestamp}.png")
    cv2.imwrite(depth_filename, depth_colormap)
    print(f"深度图像(伪彩色)已保存: {depth_filename}")
    
    # 保存原始深度数据（可选）
    depth_raw_filename = os.path.join(output_dir, f"depth_raw_{timestamp}.npy")
    np.save(depth_raw_filename, depth_image)
    print(f"原始深度数据已保存: {depth_raw_filename}")
    
    # # 显示图像（可选）
    # cv2.imshow('Color Image', color_image)
    # cv2.imshow('Depth Image', depth_colormap)
    # cv2.waitKey(2000)  # 显示2秒
    # cv2.destroyAllWindows()
        
if __name__ == "__main__":
    save_realsense_images()