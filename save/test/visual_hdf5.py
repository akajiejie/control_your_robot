import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from tqdm import tqdm
import subprocess

def visualize_hdf5(hdf5_path, output_dir="output"):
    """
    Visualize HDF5 file content:
    1. Plot robot arm joint and gripper data curves
    2. Save camera data as video files
    3. Save tactile force data as video files
    
    Parameters:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    camera_dir = os.path.join(output_dir, "video/camera")
    tactile_dir = os.path.join(output_dir, "video/tactile")
    os.makedirs(camera_dir, exist_ok=True)
    os.makedirs(tactile_dir, exist_ok=True)
    
    # Load config.json from the same directory as the HDF5 file
    hdf5_dir = os.path.dirname(hdf5_path)
    config_path = os.path.join(hdf5_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Open HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Read robot arm data
        joints = []
        gripper = []
        if 'left_arm' in config and 'left_arm' in f:
            arm_group = f['left_arm']
            if 'joint' in config['left_arm'] and 'joint' in arm_group:
                joints = arm_group['joint'][:]
            if 'gripper' in config['left_arm'] and 'gripper' in arm_group:
                gripper = arm_group['gripper'][:]
        
        # Read camera data
        cam_head = []
        cam_wrist = []
        if 'cam_head' in config and 'cam_head' in f and 'color' in f['cam_head']:
            cam_head = f['cam_head']['color'][:]
        if 'cam_wrist' in config and 'cam_wrist' in f and 'color' in f['cam_wrist']:
            cam_wrist = f['cam_wrist']['color'][:]
        
        # Read tactile data
        tactile_data = {}
        if 'left_arm_left' in config and 'left_arm_left' in f:
            tactile_group = f['left_arm_left']
            for data_type in config['left_arm_left']:
                if data_type in tactile_group:
                    tactile_data[data_type] = tactile_group[data_type][:]
        
        # 1. Plot robot arm data curves
        if len(joints) > 0 or len(gripper) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.suptitle('Robot Arm Data Visualization', fontsize=16)
            
            # Plot joint angles
            if len(joints) > 0:
                n_frames = len(joints)
                time_steps = range(n_frames)
                
                # Plot curves for 6 joints
                labels = [f'Joint {i+1}' for i in range(6)]
                for i in range(6):
                    ax1.plot(time_steps, joints[:, i], label=labels[i])
                
                ax1.set_title('Joint Angles (radians)')
                ax1.set_ylabel('Angle (rad)')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()
            
            # Plot gripper data
            if len(gripper) > 0:
                # If no joint data, get time steps separately
                if len(joints) == 0:
                    n_frames = len(gripper)
                    time_steps = range(n_frames)
                
                ax2.plot(time_steps, gripper, color='purple', label='Gripper')
                ax2.set_title('Gripper State (normalized)')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Opening Degree')
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(os.path.join(output_dir, 'arm_data_plot.png'))
            plt.close()
            print(f"Saved robot arm data plot: {os.path.join(output_dir, 'arm_data_plot.png')}")
        
        # 视频保存函数
        def save_with_ffmpeg(frames, filename, output_path, fps=30, is_tactile=False):
            """使用FFmpeg保存视频（需要系统安装FFmpeg）"""
            if len(frames) == 0:
                return
                
            # 创建临时目录存储帧图像
            temp_dir = os.path.join(output_path, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存所有帧为PNG图像
            for i, frame in enumerate(tqdm(frames, desc=f"Saving {filename} frames")):
                if is_tactile:
                    # 处理触觉数据
                    # 归一化到0-255范围
                    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                    # 转换为uint8类型
                    normalized = normalized.astype(np.uint8)
                    # 应用颜色映射
                    colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
                    # 放大图像以便观看 (16x16 -> 256x256)
                    resized = cv2.resize(colormap, (256, 256), interpolation=cv2.INTER_NEAREST)
                    # 添加标题
                    cv2.putText(resized, f"Tactile: {filename}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{i:06d}.png"), resized)
                else:
                    # 处理相机数据
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{i:06d}.png"), frame_bgr)
            
            # 使用FFmpeg创建视频
            video_path = os.path.join(output_path, f"{filename}.mp4")
            cmd = [
                    'ffmpeg',
                    '-y',  # 覆盖现有文件
                    '-loglevel', 'error',  # 只显示错误信息
                    '-framerate', str(fps),
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-pix_fmt', 'yuv420p',
                    video_path
                ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"Saved video: {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error: {e}")
            finally:
                # 清理临时文件
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
        
        # Save head camera video
        if len(cam_head) > 0:
            save_with_ffmpeg(cam_head, 'cam_head_video', camera_dir)
        
        # Save wrist camera video
        if len(cam_wrist) > 0:
            save_with_ffmpeg(cam_wrist, 'cam_wrist_video', camera_dir)
        
        # Save tactile force videos
        for data_type, data in tactile_data.items():
            # 确保数据是16x16矩阵
            if len(data.shape) == 3 and data.shape[1] == 16 and data.shape[2] == 16:
                save_with_ffmpeg(data, f"tactile_{data_type}", tactile_dir, fps=30, is_tactile=True)
            else:
                print(f"Warning: Unexpected tactile data shape {data.shape} for {data_type}")

if __name__ == "__main__":
    # Example usage
    visualize_hdf5("save/test/0.hdf5", output_dir="save/output")