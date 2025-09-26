import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from tqdm import tqdm
import subprocess
import sys

def visualize_hdf5(hdf5_path, output_dir="output", verbose=False):
    """
    Visualize HDF5 file content:
    1. Plot robot arm joint and gripper data curves for both left and right arms
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
    
    # Load config.json from the same directory as the HDF5 file
    hdf5_dir = os.path.dirname(hdf5_path)
    config_path = os.path.join(hdf5_dir, "config.json")
    
    if not os.path.exists(config_path):
        if verbose:
            print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Open HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Read robot arm data for both arms
        left_arm_data = {'joints': [], 'gripper': []}
        right_arm_data = {'joints': [], 'gripper': []}
        
        # Read left arm data (support multiple naming conventions)
        left_arm_keys = ['left_arm', 'slave_left_arm', 'master_left_arm']
        for key in left_arm_keys:
            if key in f:
                left_arm_group = f[key]
                if 'joint' in left_arm_group:
                    left_arm_data['joints'] = left_arm_group['joint'][:]
                if 'gripper' in left_arm_group:
                    left_arm_data['gripper'] = left_arm_group['gripper'][:]
                break
        
        # Read right arm data (support multiple naming conventions)
        right_arm_keys = ['right_arm', 'slave_right_arm', 'master_right_arm']
        for key in right_arm_keys:
            if key in f:
                right_arm_group = f[key]
                if 'joint' in right_arm_group:
                    right_arm_data['joints'] = right_arm_group['joint'][:]
                if 'gripper' in right_arm_group:
                    right_arm_data['gripper'] = right_arm_group['gripper'][:]
                break
        
        # Read camera data - dynamically discover camera keys (support multiple naming conventions)
        camera_data = {}
        for key in f.keys():
            if (key.startswith('cam_') or key.startswith('camera_') or 
                key.startswith('slave_cam_') or key.startswith('master_cam_')):
                if key in f and 'color' in f[key]:
                    camera_data[key] = f[key]['color'][:]
                elif key in f and 'rgb' in f[key]:
                    camera_data[key] = f[key]['rgb'][:]
                elif key in f and 'image' in f[key]:
                    camera_data[key] = f[key]['image'][:]
        
        # Read tactile data - dynamically discover tactile keys
        tactile_data = {}
        for key in f.keys():
            if 'tactile' in key.lower() or 'force' in key.lower() or 'pressure' in key.lower():
                if key in f:
                    tactile_data[key] = f[key][:]
        
        # 1. Plot robot arm data curves with 4 subplots
        has_arm_data = (len(left_arm_data['joints']) > 0 or len(left_arm_data['gripper']) > 0 or 
                       len(right_arm_data['joints']) > 0 or len(right_arm_data['gripper']) > 0)
        
        if has_arm_data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Dual-Arm Robot Data Visualization', fontsize=16)
            
            # Determine time steps
            max_frames = max(
                len(left_arm_data['joints']), len(left_arm_data['gripper']),
                len(right_arm_data['joints']), len(right_arm_data['gripper'])
            )
            if max_frames > 0:
                time_steps = range(max_frames)
                
                # Plot 1: Left Arm Joint Angles
                if len(left_arm_data['joints']) > 0:
                    labels = [f'Joint {i+1}' for i in range(min(6, left_arm_data['joints'].shape[1]))]
                    for i in range(min(6, left_arm_data['joints'].shape[1])):
                        ax1.plot(time_steps, left_arm_data['joints'][:, i], label=labels[i])
                    
                    ax1.set_title('Left Arm Joint Angles (radians)')
                    ax1.set_ylabel('Angle (rad)')
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    ax1.legend()
                else:
                    ax1.set_title('Left Arm Joint Angles (No Data)')
                    ax1.text(0.5, 0.5, 'No joint data available', ha='center', va='center', transform=ax1.transAxes)
                
                # Plot 2: Right Arm Joint Angles
                if len(right_arm_data['joints']) > 0:
                    labels = [f'Joint {i+1}' for i in range(min(6, right_arm_data['joints'].shape[1]))]
                    for i in range(min(6, right_arm_data['joints'].shape[1])):
                        ax2.plot(time_steps, right_arm_data['joints'][:, i], label=labels[i])
                    
                    ax2.set_title('Right Arm Joint Angles (radians)')
                    ax2.set_ylabel('Angle (rad)')
                    ax2.grid(True, linestyle='--', alpha=0.7)
                    ax2.legend()
                else:
                    ax2.set_title('Right Arm Joint Angles (No Data)')
                    ax2.text(0.5, 0.5, 'No joint data available', ha='center', va='center', transform=ax2.transAxes)
                
                # Plot 3: Left Arm Gripper
                if len(left_arm_data['gripper']) > 0:
                    ax3.plot(time_steps, left_arm_data['gripper'], color='purple', label='Left Gripper')
                    ax3.set_title('Left Arm Gripper State')
                    ax3.set_xlabel('Time Step')
                    ax3.set_ylabel('Opening Degree')
                    ax3.set_ylim(0, 1.1)
                    ax3.grid(True, linestyle='--', alpha=0.7)
                    ax3.legend()
                else:
                    ax3.set_title('Left Arm Gripper (No Data)')
                    ax3.text(0.5, 0.5, 'No gripper data available', ha='center', va='center', transform=ax3.transAxes)
                
                # Plot 4: Right Arm Gripper
                if len(right_arm_data['gripper']) > 0:
                    ax4.plot(time_steps, right_arm_data['gripper'], color='orange', label='Right Gripper')
                    ax4.set_title('Right Arm Gripper State')
                    ax4.set_xlabel('Time Step')
                    ax4.set_ylabel('Opening Degree')
                    ax4.set_ylim(0, 1.1)
                    ax4.grid(True, linestyle='--', alpha=0.7)
                    ax4.legend()
                else:
                    ax4.set_title('Right Arm Gripper (No Data)')
                    ax4.text(0.5, 0.5, 'No gripper data available', ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(os.path.join(output_dir, 'dual_arm_data_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Saved dual-arm data plot: {os.path.join(output_dir, 'dual_arm_data_plot.png')}")
        
        # 视频保存函数
        def save_with_ffmpeg(frames, filename, output_path, fps=30, is_tactile=False):
            """使用FFmpeg保存视频（需要系统安装FFmpeg）"""
            if len(frames) == 0:
                return
                
            # 创建临时目录存储帧图像
            temp_dir = os.path.join(output_path, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存所有帧为PNG图像
            for i, frame in enumerate(tqdm(frames, desc=f"Saving {filename} frames", disable=not verbose)):
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
                    if frame.dtype != np.uint8:
                        # 如果数据不是uint8，进行归一化
                        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    # 检查是否需要颜色空间转换
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # 假设是RGB格式，转换为BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
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
                if verbose:
                    print(f"Saved video: {video_path}")
            except subprocess.CalledProcessError as e:
                if verbose:
                    print(f"FFmpeg error: {e}")
            finally:
                # 清理临时文件
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
        
        # Save camera videos
        for camera_name, camera_frames in camera_data.items():
            if len(camera_frames) > 0:
                save_with_ffmpeg(camera_frames, f"{camera_name}_video", camera_dir)
        
        # Save tactile force videos
        for data_type, data in tactile_data.items():
            # 确保数据是16x16矩阵
            if len(data.shape) == 3 and data.shape[1] == 16 and data.shape[2] == 16:
                os.makedirs(tactile_dir, exist_ok=True)
                save_with_ffmpeg(data, f"tactile_{data_type}", tactile_dir, fps=30, is_tactile=True)
            else:
                if verbose:
                    print(f"Warning: Unexpected tactile data shape {data.shape} for {data_type}")
        
        # Print summary
        if verbose:
            print(f"\n=== Visualization Summary ===")
            print(f"Left arm joints: {len(left_arm_data['joints'])} frames")
            print(f"Left arm gripper: {len(left_arm_data['gripper'])} frames")
            print(f"Right arm joints: {len(right_arm_data['joints'])} frames")
            print(f"Right arm gripper: {len(right_arm_data['gripper'])} frames")
            print(f"Camera data: {len(camera_data)} cameras")
            print(f"Tactile data: {len(tactile_data)} sensors")

def explore_hdf5_structure(hdf5_path, verbose=False):
    """
    Explore and print the structure of HDF5 file
    
    Parameters:
        hdf5_path: Path to HDF5 file
    """
    if not verbose:
        return
    print(f"=== HDF5 Structure: {hdf5_path} ===")
    with h5py.File(hdf5_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        f.visititems(print_structure)

def visualize_folder(folder_path, output_base_dir="output", verbose=False):
    """
    可视化文件夹下的所有HDF5文件
    
    Parameters:
        folder_path: 包含HDF5文件的文件夹路径
        output_base_dir: 输出基础目录
    """
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 查找所有HDF5文件
    hdf5_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.hdf5') or file.endswith('.h5'):
            hdf5_files.append(os.path.join(folder_path, file))
    
    if not hdf5_files:
        print(f"在文件夹 {folder_path} 中未找到HDF5文件")
        return
    
    # Quiet mode: only print count; verbose: also list files
    if verbose:
        print(f"找到 {len(hdf5_files)} 个HDF5文件:")
        for file in hdf5_files:
            print(f"  - {os.path.basename(file)}")
    else:
        print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 跟踪处理结果
    successful_files = []
    failed_files = []
    
    # 处理每个HDF5文件（显示总体进度条）
    with tqdm(total=len(hdf5_files), desc="Processing HDF5 files", unit="file", disable=False) as pbar:
        for i, hdf5_file in enumerate(hdf5_files, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"处理文件 {i}/{len(hdf5_files)}: {os.path.basename(hdf5_file)}")
                print(f"{'='*60}")
            
            # 为每个文件创建独立的输出目录
            file_name = os.path.splitext(os.path.basename(hdf5_file))[0]
            output_dir = os.path.join(output_base_dir, file_name)
            
            try:
                # 首先探索文件结构
                explore_hdf5_structure(hdf5_file, verbose=verbose)
                if verbose:
                    print("\n" + "-"*50 + "\n")
                
                # 然后可视化数据
                visualize_hdf5(hdf5_file, output_dir, verbose=verbose)
                
                # 记录成功处理的文件
                successful_files.append(os.path.basename(hdf5_file))
                
                if verbose:
                    print(f"✓ 文件 {os.path.basename(hdf5_file)} 处理完成")
                
            except Exception as e:
                # 记录失败的文件
                failed_files.append(os.path.basename(hdf5_file))
                print(f"✗ 处理文件 {os.path.basename(hdf5_file)} 时出错: {str(e)}")
            finally:
                pbar.update(1)
    
    # 输出处理统计结果
    total_files = len(hdf5_files)
    successful_count = len(successful_files)
    failed_count = len(failed_files)
    success_rate = (successful_count / total_files * 100) if total_files > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*60}")
    print(f"\n📊 处理统计结果:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {successful_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"成功率: {success_rate:.1f}%")
    
    if failed_files:
        print(f"\n❌ 处理失败的文件:")
        for i, failed_file in enumerate(failed_files, 1):
            print(f"  {i}. {failed_file}")
    else:
        print(f"\n✅ 所有文件处理成功！")
    
    print(f"{'='*60}")

def get_hdf5_files_info(folder_path):
    """
    获取文件夹中所有HDF5文件的信息
    
    Parameters:
        folder_path: 文件夹路径
        
    Returns:
        list: 包含文件信息的列表
    """
    if not os.path.exists(folder_path):
        return []
    
    files_info = []
    for file in os.listdir(folder_path):
        if file.endswith('.hdf5') or file.endswith('.h5'):
            file_path = os.path.join(folder_path, file)
            file_size = os.path.getsize(file_path)
            
            # 获取文件基本信息
            info = {
                'name': file,
                'path': file_path,
                'size_mb': file_size / (1024 * 1024),
                'structure': {}
            }
            
            # 获取HDF5文件结构信息
            try:
                with h5py.File(file_path, 'r') as f:
                    def collect_structure(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            info['structure'][name] = {
                                'shape': obj.shape,
                                'dtype': str(obj.dtype)
                            }
                        elif isinstance(obj, h5py.Group):
                            info['structure'][name] = {'type': 'group'}
                    
                    f.visititems(collect_structure)
            except Exception as e:
                info['error'] = str(e)
            
            files_info.append(info)
    
    return files_info

def print_files_summary(files_info, verbose=False):
    """
    打印文件信息摘要
    
    Parameters:
        files_info: 文件信息列表
    """
    if not verbose:
        return
    if not files_info:
        print("没有找到HDF5文件")
        return
    print(f"\n=== HDF5文件摘要 ===")
    print(f"共找到 {len(files_info)} 个HDF5文件:\n")
    for i, info in enumerate(files_info, 1):
        print(f"{i}. {info['name']}")
        print(f"   大小: {info['size_mb']:.2f} MB")
        if 'error' in info:
            print(f"   状态: 错误 - {info['error']}")
        else:
            print(f"   状态: 正常")
            print(f"   结构:")
            for key, value in info['structure'].items():
                if isinstance(value, dict) and 'shape' in value:
                    print(f"     - {key}: {value['shape']} ({value['dtype']})")
                else:
                    print(f"     - {key}: {value}")
        print()

if __name__ == "__main__":
    # 文件夹路径
    folder_path = "/home/usst/kwj/GitCode/control_your_robot_jie/test/pick_place_cup/"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        print("请检查路径是否正确")
        sys.exit(1)
    
    # 获取文件信息并仅输出数量（安静模式）
    files_info = get_hdf5_files_info(folder_path)
    print(f"找到 {len(files_info)} 个HDF5文件")
    
    # 直接批量处理（启用详细输出以便调试）
    if files_info:
        output_dir = "save/output/test/pick_place_cup/"
        visualize_folder(folder_path, output_dir, verbose=True)
    else:
        print("没有找到HDF5文件，无法处理")