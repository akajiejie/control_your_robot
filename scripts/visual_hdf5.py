import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import json
from tqdm import tqdm
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for video generation
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import shutil


def visualize_hdf5(hdf5_path, output_dir="output", verbose=False, force_regenerate=False):
    """
    Visualize HDF5 file content:
    1. Create synchronized videos combining camera feeds with dynamic robot data plots
    2. Save tactile force data as video files
    3. Support for eefort data visualization
    
    Parameters:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory
        verbose: Enable verbose output
        force_regenerate: Force regeneration even if output exists
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    camera_dir = os.path.join(output_dir, "video/camera")
    tactile_dir = os.path.join(output_dir, "video/tactile")
    combined_dir = os.path.join(output_dir, "video/combined")
    os.makedirs(camera_dir, exist_ok=True)
    
    # Check if processing is needed (caching mechanism)
    def should_process_file():
        if force_regenerate:
            return True
            
        # Check if HDF5 file is newer than existing output
        if not os.path.exists(hdf5_path):
            return False
            
        hdf5_mtime = os.path.getmtime(hdf5_path)
        
        # Check if any video files exist and are newer than HDF5
        video_dirs = [camera_dir, combined_dir]
        if os.path.exists(tactile_dir):
            video_dirs.append(tactile_dir)
            
        for video_dir in video_dirs:
            if os.path.exists(video_dir):
                for file in os.listdir(video_dir):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(video_dir, file)
                        if os.path.getmtime(video_path) > hdf5_mtime:
                            if verbose:
                                print(f"Skipping {os.path.basename(hdf5_path)} - videos are up to date")
                            return False
        return True
    
    if not should_process_file():
        return
    
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
        
        # Read eefort data for both arms
        eefort_data = {'left': [], 'right': []}
        
        # Check within arm groups for eefort data
        for key in left_arm_keys:
            if key in f:
                arm_group = f[key]
                if 'eefort' in arm_group:
                    eefort_data['left'] = arm_group['eefort'][:]
                    break
        
        for key in right_arm_keys:
            if key in f:
                arm_group = f[key]
                if 'eefort' in arm_group:
                    eefort_data['right'] = arm_group['eefort'][:]
                    break
        
        # Also check for standalone eefort data
        for key in f.keys():
            if 'eefort' in key.lower() or 'effort' in key.lower() or 'force' in key.lower():
                if 'left' in key.lower():
                    eefort_data['left'] = f[key][:]
                elif 'right' in key.lower():
                    eefort_data['right'] = f[key][:]
        
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
        
        # Pre-calculate data statistics for consistent plot ranges
        def calculate_data_ranges(left_arm_data, right_arm_data, eefort_data):
            """Calculate min/max ranges for all data to ensure consistent plot scaling"""
            ranges = {
                'left_joints': {'min': 0, 'max': 1, 'has_data': False},
                'right_joints': {'min': 0, 'max': 1, 'has_data': False},
                'left_gripper': {'min': 0, 'max': 1, 'has_data': False},
                'right_gripper': {'min': 0, 'max': 1, 'has_data': False},
                'left_eefort': {'min': 0, 'max': 1, 'has_data': False},
                'right_eefort': {'min': 0, 'max': 1, 'has_data': False},
                'max_frames': 0
            }
            
            # Calculate joint ranges
            if len(left_arm_data['joints']) > 0:
                ranges['left_joints']['min'] = float(np.min(left_arm_data['joints']))
                ranges['left_joints']['max'] = float(np.max(left_arm_data['joints']))
                ranges['left_joints']['has_data'] = True
                ranges['max_frames'] = max(ranges['max_frames'], len(left_arm_data['joints']))
            
            if len(right_arm_data['joints']) > 0:
                ranges['right_joints']['min'] = float(np.min(right_arm_data['joints']))
                ranges['right_joints']['max'] = float(np.max(right_arm_data['joints']))
                ranges['right_joints']['has_data'] = True
                ranges['max_frames'] = max(ranges['max_frames'], len(right_arm_data['joints']))
            
            # Calculate gripper ranges
            if len(left_arm_data['gripper']) > 0:
                ranges['left_gripper']['min'] = float(np.min(left_arm_data['gripper']))
                ranges['left_gripper']['max'] = float(np.max(left_arm_data['gripper']))
                ranges['left_gripper']['has_data'] = True
                ranges['max_frames'] = max(ranges['max_frames'], len(left_arm_data['gripper']))
            
            if len(right_arm_data['gripper']) > 0:
                ranges['right_gripper']['min'] = float(np.min(right_arm_data['gripper']))
                ranges['right_gripper']['max'] = float(np.max(right_arm_data['gripper']))
                ranges['right_gripper']['has_data'] = True
                ranges['max_frames'] = max(ranges['max_frames'], len(right_arm_data['gripper']))
            
            # Calculate eefort ranges
            if len(eefort_data['left']) > 0:
                ranges['left_eefort']['min'] = float(np.min(eefort_data['left']))
                ranges['left_eefort']['max'] = float(np.max(eefort_data['left']))
                ranges['left_eefort']['has_data'] = True
                ranges['max_frames'] = max(ranges['max_frames'], len(eefort_data['left']))
            
            if len(eefort_data['right']) > 0:
                ranges['right_eefort']['min'] = float(np.min(eefort_data['right']))
                ranges['right_eefort']['max'] = float(np.max(eefort_data['right']))
                ranges['right_eefort']['has_data'] = True
                ranges['max_frames'] = max(ranges['max_frames'], len(eefort_data['right']))
            
            # Add some padding to ranges for better visualization
            for key in ['left_joints', 'right_joints', 'left_eefort', 'right_eefort']:
                if ranges[key]['has_data']:
                    data_range = ranges[key]['max'] - ranges[key]['min']
                    padding = data_range * 0.1 if data_range > 0 else 0.1
                    ranges[key]['min'] -= padding
                    ranges[key]['max'] += padding
            
            # Gripper range is typically 0-1, but add some padding
            for key in ['left_gripper', 'right_gripper']:
                if ranges[key]['has_data']:
                    ranges[key]['min'] = max(0, ranges[key]['min'] - 0.05)
                    ranges[key]['max'] = min(1, ranges[key]['max'] + 0.05)
            
            return ranges

        # Create synchronized videos combining camera and robot data
        def create_dynamic_plot_frame(frame_idx, data_ranges, left_arm_data, right_arm_data, eefort_data, fig=None, axes=None):
            """Create a single frame of the dynamic plot showing robot data up to current frame"""
            # Reuse figure and axes if provided (major performance improvement)
            if fig is None or axes is None:
                # Set seaborn style only once
                sns.set_style("whitegrid")
                plt.style.use('seaborn-v0_8')
                
                # Create figure with subplots
                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
                axes = {
                    'ax1': fig.add_subplot(gs[0, 0]),
                    'ax2': fig.add_subplot(gs[0, 1]),
                    'ax3': fig.add_subplot(gs[1, :]),
                    'ax4': fig.add_subplot(gs[2, :])
                }
            else:
                # Clear existing plots
                for ax in axes.values():
                    ax.clear()
            
            # Current time range for plotting
            max_frames = data_ranges['max_frames']
            current_range = min(frame_idx + 1, max_frames)
            time_steps = np.arange(current_range)
            full_time_steps = np.arange(max_frames)  # For consistent x-axis
            
            # Plot 1: Left Arm Joint Angles
            ax1 = axes['ax1']
            if data_ranges['left_joints']['has_data'] and current_range > 0:
                palette = sns.color_palette("husl", min(6, left_arm_data['joints'].shape[1]))
                for i in range(min(6, left_arm_data['joints'].shape[1])):
                    ax1.plot(time_steps, left_arm_data['joints'][:current_range, i], 
                            label=f'Joint {i+1}', color=palette[i], linewidth=2)
                    # Highlight current point
                    if frame_idx < len(left_arm_data['joints']):
                        ax1.scatter(frame_idx, left_arm_data['joints'][frame_idx, i], 
                                  color=palette[i], s=50, zorder=5)
                
                # Set fixed axis ranges
                ax1.set_xlim(0, max_frames - 1)
                ax1.set_ylim(data_ranges['left_joints']['min'], data_ranges['left_joints']['max'])
                ax1.set_title('Left Arm Joint Angles', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Angle (rad)')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No Left Arm Joint Data', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12)
                ax1.set_title('Left Arm Joint Angles', fontsize=14, fontweight='bold')
                ax1.set_xlim(0, max_frames - 1 if max_frames > 0 else 1)
                ax1.set_ylim(0, 1)
            
            # Plot 2: Right Arm Joint Angles
            ax2 = axes['ax2']
            if data_ranges['right_joints']['has_data'] and current_range > 0:
                palette = sns.color_palette("husl", min(6, right_arm_data['joints'].shape[1]))
                for i in range(min(6, right_arm_data['joints'].shape[1])):
                    ax2.plot(time_steps, right_arm_data['joints'][:current_range, i], 
                            label=f'Joint {i+1}', color=palette[i], linewidth=2)
                    # Highlight current point
                    if frame_idx < len(right_arm_data['joints']):
                        ax2.scatter(frame_idx, right_arm_data['joints'][frame_idx, i], 
                                  color=palette[i], s=50, zorder=5)
                
                # Set fixed axis ranges
                ax2.set_xlim(0, max_frames - 1)
                ax2.set_ylim(data_ranges['right_joints']['min'], data_ranges['right_joints']['max'])
                ax2.set_title('Right Arm Joint Angles', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Angle (rad)')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Right Arm Joint Data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Right Arm Joint Angles', fontsize=14, fontweight='bold')
                ax2.set_xlim(0, max_frames - 1 if max_frames > 0 else 1)
                ax2.set_ylim(0, 1)
            
            # Plot 3: Gripper States
            ax3 = axes['ax3']
            has_gripper_data = data_ranges['left_gripper']['has_data'] or data_ranges['right_gripper']['has_data']
            
            if has_gripper_data:
                # Calculate combined gripper range
                gripper_min = min(data_ranges['left_gripper']['min'] if data_ranges['left_gripper']['has_data'] else 1,
                                data_ranges['right_gripper']['min'] if data_ranges['right_gripper']['has_data'] else 1)
                gripper_max = max(data_ranges['left_gripper']['max'] if data_ranges['left_gripper']['has_data'] else 0,
                                data_ranges['right_gripper']['max'] if data_ranges['right_gripper']['has_data'] else 0)
                
                if data_ranges['left_gripper']['has_data'] and current_range > 0:
                    ax3.plot(time_steps, left_arm_data['gripper'][:current_range], 
                            color='purple', label='Left Gripper', linewidth=3)
                    if frame_idx < len(left_arm_data['gripper']):
                        ax3.scatter(frame_idx, left_arm_data['gripper'][frame_idx], 
                                  color='purple', s=60, zorder=5)
                
                if data_ranges['right_gripper']['has_data'] and current_range > 0:
                    ax3.plot(time_steps, right_arm_data['gripper'][:current_range], 
                            color='orange', label='Right Gripper', linewidth=3)
                    if frame_idx < len(right_arm_data['gripper']):
                        ax3.scatter(frame_idx, right_arm_data['gripper'][frame_idx], 
                                  color='orange', s=60, zorder=5)
                
                # Set fixed axis ranges
                ax3.set_xlim(0, max_frames - 1)
                ax3.set_ylim(gripper_min, gripper_max)
                ax3.set_title('Gripper States', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Opening Degree')
                ax3.legend(fontsize=12)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No Gripper Data', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Gripper States', fontsize=14, fontweight='bold')
                ax3.set_xlim(0, max_frames - 1 if max_frames > 0 else 1)
                ax3.set_ylim(0, 1)
            
            # Plot 4: Eefort Data
            ax4 = axes['ax4']
            has_eefort_data = data_ranges['left_eefort']['has_data'] or data_ranges['right_eefort']['has_data']
            
            if has_eefort_data:
                # Calculate combined eefort range
                eefort_min = min(data_ranges['left_eefort']['min'] if data_ranges['left_eefort']['has_data'] else 0,
                               data_ranges['right_eefort']['min'] if data_ranges['right_eefort']['has_data'] else 0)
                eefort_max = max(data_ranges['left_eefort']['max'] if data_ranges['left_eefort']['has_data'] else 1,
                               data_ranges['right_eefort']['max'] if data_ranges['right_eefort']['has_data'] else 1)
                
                if data_ranges['left_eefort']['has_data'] and current_range > 0:
                    # If multi-dimensional, plot each component
                    if len(eefort_data['left'].shape) > 1:
                        palette = sns.color_palette("Reds_r", eefort_data['left'].shape[1])
                        for i in range(min(6, eefort_data['left'].shape[1])):
                            ax4.plot(time_steps, eefort_data['left'][:current_range, i], 
                                    color=palette[i], label=f'Left F{i+1}', linewidth=2)
                    else:
                        ax4.plot(time_steps, eefort_data['left'][:current_range], 
                                color='red', label='Left Force', linewidth=3)
                
                if data_ranges['right_eefort']['has_data'] and current_range > 0:
                    # If multi-dimensional, plot each component
                    if len(eefort_data['right'].shape) > 1:
                        palette = sns.color_palette("Blues_r", eefort_data['right'].shape[1])
                        for i in range(min(6, eefort_data['right'].shape[1])):
                            ax4.plot(time_steps, eefort_data['right'][:current_range, i], 
                                    color=palette[i], label=f'Right F{i+1}', linewidth=2)
                    else:
                        ax4.plot(time_steps, eefort_data['right'][:current_range], 
                                color='blue', label='Right Force', linewidth=3)
                
                # Set fixed axis ranges
                ax4.set_xlim(0, max_frames - 1)
                ax4.set_ylim(eefort_min, eefort_max)
                ax4.set_title('Joint-Effector Forces', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Force (N)')
                ax4.set_xlabel('Time Step')
                ax4.legend(fontsize=10, ncol=2)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Force Data', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Joint-Effector Forces', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Time Step')
                ax4.set_xlim(0, max_frames - 1 if max_frames > 0 else 1)
                ax4.set_ylim(0, 1)
            
            # Add frame information
            fig.suptitle(f'Robot Data Visualization - Frame {frame_idx + 1}/{max_frames}', 
                        fontsize=16, fontweight='bold')
            
            # Convert plot to image
            fig.canvas.draw()
            # Try newer method first, fall back to older method if needed
            try:
                buf = fig.canvas.buffer_rgba()
                plot_img = np.asarray(buf)[:, :, :3]  # Remove alpha channel
            except AttributeError:
                try:
                    buf = fig.canvas.tostring_rgb()
                    plot_img = np.frombuffer(buf, dtype=np.uint8)
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                except AttributeError:
                    # For newer matplotlib versions
                    buf = fig.canvas.renderer.buffer_rgba()
                    plot_img = np.asarray(buf)[:, :, :3]  # Remove alpha channel
            # Don't close the figure if we're reusing it
            if axes is None:
                plt.close(fig)
            
            return plot_img, fig, axes
        
        def create_combined_video(camera_frames, camera_name, left_arm_data, right_arm_data, eefort_data, output_path, fps=30):
            """Create synchronized video combining camera feed and dynamic plots"""
            if len(camera_frames) == 0:
                return
                
            # Calculate data ranges for consistent plotting
            data_ranges = calculate_data_ranges(left_arm_data, right_arm_data, eefort_data)
            
            # Determine the number of frames
            max_frames = len(camera_frames)
            
            # Create temporary directory for combined frames
            temp_dir = os.path.join(output_path, f"temp_{camera_name}_combined")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Initialize reusable figure and axes for better performance
                fig, axes = None, None
                
                # Generate combined frames
                for frame_idx in tqdm(range(max_frames), desc=f"Creating {camera_name} combined frames", disable=not verbose):
                    # Get camera frame
                    camera_frame = camera_frames[frame_idx]
                    
                    # Process camera frame
                    if camera_frame.dtype != np.uint8:
                        camera_frame = cv2.normalize(camera_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    if len(camera_frame.shape) == 3 and camera_frame.shape[2] == 3:
                        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
                    
                    # Resize camera frame to standard size
                    camera_height, camera_width = 480, 640
                    camera_frame_resized = cv2.resize(camera_frame, (camera_width, camera_height))
                    
                    # Generate plot frame (reuse figure and axes)
                    plot_img, fig, axes = create_dynamic_plot_frame(frame_idx, data_ranges, left_arm_data, right_arm_data, eefort_data, fig, axes)
                    plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
                    
                    # Resize plot to match camera height
                    plot_aspect_ratio = plot_img.shape[1] / plot_img.shape[0]
                    plot_width = int(camera_height * plot_aspect_ratio)
                    plot_img_resized = cv2.resize(plot_img_bgr, (plot_width, camera_height))
                    
                    # Combine camera and plot horizontally
                    combined_frame = np.hstack([camera_frame_resized, plot_img_resized])
                    
                    # Add title overlay
                    title_text = f"{camera_name} - Frame {frame_idx + 1}/{max_frames}"
                    cv2.putText(combined_frame, title_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Save frame
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{frame_idx:06d}.png"), combined_frame)
                
                # Create video using FFmpeg
                video_path = os.path.join(output_path, f"{camera_name}_with_plots.mp4")
                cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-framerate', str(fps),
                    '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                    '-pix_fmt', 'yuv420p', video_path
                ]
                
                subprocess.run(cmd, check=True)
                if verbose:
                    print(f"Saved combined video: {video_path}")
                    
            except Exception as e:
                if verbose:
                    print(f"Error creating combined video for {camera_name}: {e}")
            finally:
                # Close the reused figure
                if fig is not None:
                    plt.close(fig)
                
                # Clean up temporary files
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
        
        # 优化的视频保存函数
        def save_with_ffmpeg_optimized(frames, filename, output_path, fps=30, is_tactile=False):
            """使用FFmpeg保存视频（优化版本，减少磁盘I/O）"""
            if len(frames) == 0:
                return
                
            video_path = os.path.join(output_path, f"{filename}.mp4")
            
            # 使用FFmpeg的stdin管道直接传输帧数据，避免临时文件
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖现有文件
                '-loglevel', 'error',  # 只显示错误信息
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '640x480' if not is_tactile else '256x256',  # 设置帧大小
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',  # 从stdin读取
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'fast',  # 使用更快的预设
                '-pix_fmt', 'yuv420p',
                video_path
            ]
            
            process = None
            stdin_closed = False
            
            try:
                # 启动FFmpeg进程
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 直接向FFmpeg管道写入帧数据
                for i, frame in enumerate(tqdm(frames, desc=f"Encoding {filename}", disable=not verbose)):
                    # 检查进程是否还在运行
                    if process.poll() is not None:
                        if verbose:
                            print(f"FFmpeg process terminated early for {filename}")
                        break
                    
                    if is_tactile:
                        # 处理触觉数据
                        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
                        resized = cv2.resize(colormap, (256, 256), interpolation=cv2.INTER_NEAREST)
                        cv2.putText(resized, f"Tactile: {filename}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        processed_frame = resized
                    else:
                        # 处理相机数据
                        if frame.dtype != np.uint8:
                            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame
                        
                        # 确保帧大小一致
                        processed_frame = cv2.resize(frame_bgr, (640, 480))
                    
                    # 写入帧数据到FFmpeg
                    try:
                        if process.stdin and not stdin_closed:
                            process.stdin.write(processed_frame.tobytes())
                            process.stdin.flush()  # 确保数据被写入
                    except (BrokenPipeError, OSError) as e:
                        if verbose:
                            print(f"Pipe broken while writing frame {i} for {filename}: {e}")
                        break
                
                # 安全关闭stdin
                try:
                    if process and process.stdin and not stdin_closed:
                        process.stdin.close()
                        stdin_closed = True
                except (BrokenPipeError, OSError):
                    # stdin已经被关闭，忽略错误
                    stdin_closed = True
                
                # 等待FFmpeg完成
                if process:
                    try:
                        stdout_output, stderr_output = process.communicate(timeout=30)
                        
                        if process.returncode == 0:
                            if verbose:
                                print(f"Saved video: {video_path}")
                        else:
                            if verbose:
                                print(f"FFmpeg error for {filename}: {stderr_output.decode() if stderr_output else 'Unknown error'}")
                    except subprocess.TimeoutExpired:
                        if verbose:
                            print(f"FFmpeg timeout for {filename}, terminating process")
                        process.kill()
                        process.communicate()
                        
            except Exception as e:
                if verbose:
                    print(f"Error creating video {filename}: {e}")
            finally:
                # 确保进程被正确清理
                if process:
                    try:
                        if not stdin_closed and process.stdin:
                            process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                    
                    # 如果进程还在运行，终止它
                    if process.poll() is None:
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        except:
                            pass
        
        # Determine maximum frames for synchronization
        max_frames = 0
        if camera_data:
            max_frames = max(len(frames) for frames in camera_data.values())
        
        # Save combined camera videos with dynamic plots
        combined_dir = os.path.join(output_dir, "video/combined")
        os.makedirs(combined_dir, exist_ok=True)
        
        for camera_name, camera_frames in camera_data.items():
            if len(camera_frames) > 0:
                # Create combined video with plots
                create_combined_video(camera_frames, camera_name, left_arm_data, right_arm_data, 
                                    eefort_data, combined_dir, fps=30)
                
                # Also save original camera video for reference
                save_with_ffmpeg_optimized(camera_frames, f"{camera_name}_video", camera_dir)
        
        # Save tactile force videos
        for data_type, data in tactile_data.items():
            # 确保数据是16x16矩阵
            if len(data.shape) == 3 and data.shape[1] == 16 and data.shape[2] == 16:
                os.makedirs(tactile_dir, exist_ok=True)
                save_with_ffmpeg_optimized(data, f"tactile_{data_type}", tactile_dir, fps=30, is_tactile=True)
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
            print(f"Left arm eefort: {len(eefort_data['left'])} frames")
            print(f"Right arm eefort: {len(eefort_data['right'])} frames")
            print(f"Camera data: {len(camera_data)} cameras")
            print(f"Tactile data: {len(tactile_data)} sensors")
            print(f"Generated combined videos: {len([name for name, frames in camera_data.items() if len(frames) > 0])}")
            print(f"Output directories:")
            print(f"  - Combined videos: {combined_dir}")
            print(f"  - Original camera videos: {camera_dir}")
            if tactile_data:
                print(f"  - Tactile videos: {tactile_dir}")

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

def process_single_hdf5(args):
    """
    处理单个HDF5文件的函数，用于多进程处理
    
    Parameters:
        args: (hdf5_file, output_base_dir, verbose, force_regenerate) 元组
        
    Returns:
        tuple: (success, filename, error_message, skipped)
    """
    hdf5_file, output_base_dir, verbose, force_regenerate = args
    file_name = os.path.splitext(os.path.basename(hdf5_file))[0]
    output_dir = os.path.join(output_base_dir, file_name)
    
    try:
        # 首先探索文件结构
        explore_hdf5_structure(hdf5_file, verbose=verbose)
        
        # 直接可视化数据，简化逻辑
        visualize_hdf5(hdf5_file, output_dir, verbose=verbose, force_regenerate=force_regenerate)
        
        return True, os.path.basename(hdf5_file), None, False
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return False, os.path.basename(hdf5_file), error_msg, False

def visualize_folder(folder_path, output_base_dir="output", verbose=False, max_workers=None, force_regenerate=False):
    """
    可视化文件夹下的所有HDF5文件（支持多进程加速和智能缓存）
    
    Parameters:
        folder_path: 包含HDF5文件的文件夹路径
        output_base_dir: 输出基础目录
        verbose: 详细输出模式
        max_workers: 最大工作进程数，None表示使用CPU核心数
        force_regenerate: 强制重新生成所有视频
    # 快速处理（推荐）
    python visual_hdf5.py /path/to/hdf5/folder -v

    # 最大性能（多核系统）
    python visual_hdf5.py /path/to/hdf5/folder -j 8 -v

    # 强制重新生成
    python visual_hdf5.py /path/to/hdf5/folder -f -v
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
    
    # 确定工作进程数
    if max_workers is None or max_workers == 0:
        max_workers = min(cpu_count(), len(hdf5_files))
    else:
        max_workers = min(max_workers, len(hdf5_files))
    
    # 确保至少有1个进程
    max_workers = max(1, max_workers)
    
    # Quiet mode: only print count; verbose: also list files
    if verbose:
        print(f"找到 {len(hdf5_files)} 个HDF5文件:")
        for file in hdf5_files:
            print(f"  - {os.path.basename(file)}")
    else:
        print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    print(f"使用 {max_workers} 个进程并行处理")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 跟踪处理结果
    successful_files = []
    failed_files = []
    skipped_files = []
    
    # 记录开始时间
    start_time = time.time()
    
    # 准备参数
    args_list = [(hdf5_file, output_base_dir, verbose, force_regenerate) for hdf5_file in hdf5_files]
    
    # 使用多进程处理
    if max_workers > 1:
        try:
            with Pool(processes=max_workers) as pool:
                # 使用imap_unordered获得进度反馈
                results = []
                with tqdm(total=len(hdf5_files), desc="Processing HDF5 files", unit="file") as pbar:
                    for result in pool.imap_unordered(process_single_hdf5, args_list):
                        results.append(result)
                        success, filename, error, skipped = result
                        if success:
                            if skipped:
                                skipped_files.append(filename)
                                if verbose:
                                    print(f"⏭ 文件 {filename} 已跳过（缓存有效）")
                            else:
                                successful_files.append(filename)
                                if verbose:
                                    print(f"✓ 文件 {filename} 处理完成")
                        else:
                            failed_files.append(filename)
                            print(f"✗ 处理文件 {filename} 时出错: {error}")
                        pbar.update(1)
        except Exception as e:
            print(f"多进程处理出错，切换到单进程模式: {e}")
            max_workers = 1
    
    if max_workers == 1:
        # 单进程处理（用于调试或多进程失败时的回退）
        print("使用单进程处理模式")
        with tqdm(total=len(hdf5_files), desc="Processing HDF5 files", unit="file") as pbar:
            for args in args_list:
                success, filename, error, skipped = process_single_hdf5(args)
                if success:
                    if skipped:
                        skipped_files.append(filename)
                        if verbose:
                            print(f"⏭ 文件 {filename} 已跳过（缓存有效）")
                    else:
                        successful_files.append(filename)
                        if verbose:
                            print(f"✓ 文件 {filename} 处理完成")
                else:
                    failed_files.append(filename)
                    print(f"✗ 处理文件 {filename} 时出错: {error}")
                pbar.update(1)
    
    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 输出处理统计结果
    total_files = len(hdf5_files)
    successful_count = len(successful_files)
    failed_count = len(failed_files)
    skipped_count = len(skipped_files)
    processed_count = successful_count + failed_count
    success_rate = (successful_count / total_files * 100) if total_files > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！")
    print(f"输出目录: {output_base_dir}")
    print(f"处理时间: {processing_time:.2f} 秒")
    if processed_count > 0:
        print(f"平均每处理文件: {processing_time/processed_count:.2f} 秒")
    print(f"{'='*60}")
    print(f"\n📊 处理统计结果:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {successful_count} 个文件")
    print(f"跳过文件: {skipped_count} 个文件（缓存有效）")
    print(f"处理失败: {failed_count} 个文件")
    print(f"成功率: {success_rate:.1f}%")
    print(f"使用进程数: {max_workers}")
    if skipped_count > 0:
        print(f"缓存节省时间: 约 {skipped_count * (processing_time/max(processed_count, 1)):.1f} 秒")
    
    if failed_files:
        print(f"\n❌ 处理失败的文件:")
        for i, failed_file in enumerate(failed_files, 1):
            print(f"  {i}. {failed_file}")
    
    if skipped_files and verbose:
        print(f"\n⏭ 跳过的文件（缓存有效）:")
        for i, skipped_file in enumerate(skipped_files, 1):
            print(f"  {i}. {skipped_file}")
    
    if successful_count == total_files:
        print(f"\n✅ 所有文件处理成功！")
    elif successful_count + skipped_count == total_files:
        print(f"\n✅ 所有文件完成（包括缓存）！")
    
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
    # 多进程安全保护
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='高性能HDF5数据可视化工具')
    parser.add_argument('input_path', help='输入HDF5文件或包含HDF5文件的文件夹路径')
    parser.add_argument('-o', '--output', default='save/output/test/feed_test/', 
                       help='输出目录 (默认: save/output/test/feed_test/)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='启用详细输出')
    parser.add_argument('-j', '--jobs', type=int, default=8,
                       help='并行处理的进程数 (默认: 1, 使用-j 0自动检测CPU核心数)')
    parser.add_argument('-f', '--force', action='store_true',
                       help='强制重新生成所有视频，忽略缓存')
    parser.add_argument('--single-file', action='store_true',
                       help='处理单个HDF5文件而不是文件夹')
    
    args = parser.parse_args()
    
    
    # 处理进程数参数
    if args.jobs == 0:
        args.jobs = cpu_count()
    elif args.jobs < 0:
        args.jobs = 1
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input_path):
        print(f"路径不存在: {args.input_path}")
        print("请检查路径是否正确")
        sys.exit(1)
    
    # 处理单个文件
    if args.single_file or args.input_path.endswith(('.hdf5', '.h5')):
        if not args.input_path.endswith(('.hdf5', '.h5')):
            print("错误: 指定了 --single-file 但输入不是HDF5文件")
            sys.exit(1)
        
        print(f"处理单个HDF5文件: {args.input_path}")
        
        # 为单个文件创建输出目录
        file_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_dir = os.path.join(args.output, file_name)
        
        try:
            start_time = time.time()
            explore_hdf5_structure(args.input_path, verbose=args.verbose)
            visualize_hdf5(args.input_path, output_dir, verbose=args.verbose, 
                          force_regenerate=args.force)
            end_time = time.time()
            
            print(f"\n✅ 文件处理完成！")
            print(f"处理时间: {end_time - start_time:.2f} 秒")
            print(f"输出目录: {output_dir}")
            
        except Exception as e:
            print(f"✗ 处理文件时出错: {e}")
            sys.exit(1)
    
    # 处理文件夹
    else:
        # 获取文件信息
        files_info = get_hdf5_files_info(args.input_path)
        
        if not files_info:
            print(f"在文件夹 {args.input_path} 中没有找到HDF5文件")
            sys.exit(1)
        
        print(f"找到 {len(files_info)} 个HDF5文件")
        
        # 显示性能优化信息
        if not args.force:
            print("🚀 性能优化已启用:")
            print("  - 智能缓存: 跳过已生成的视频")
            print("  - 多进程处理: 并行处理多个文件")
            print("  - 图形重用: 减少matplotlib开销")
            print("  - 管道编码: 减少磁盘I/O")
            if args.jobs:
                print(f"  - 使用 {args.jobs} 个进程")
            else:
                print(f"  - 使用 {min(cpu_count(), len(files_info))} 个进程")
        else:
            print("⚠️  强制重新生成模式: 将忽略所有缓存")
        
        # 批量处理
        visualize_folder(args.input_path, args.output, verbose=args.verbose, 
                        max_workers=args.jobs, force_regenerate=args.force)