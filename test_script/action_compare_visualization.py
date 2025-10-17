import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from tqdm import tqdm
from collections import defaultdict
import glob


def read_joint_data_from_hdf5(hdf5_path, verbose=False):
    """
    从HDF5文件中读取joint数据
    
    Parameters:
        hdf5_path: HDF5文件路径
        verbose: 是否输出详细信息
        
    Returns:
        dict: 包含left_arm和right_arm的joint数据
    """
    joint_data = {'left_arm': [], 'right_arm': []}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Read left arm data (support multiple naming conventions and data structures)
            left_arm_paths = [
                # 直接路径格式
                'left_arm/joint',
                'slave_left_arm/joint', 
                'master_left_arm/joint',
                # observations路径格式
                'observations/left_arm/joint',
                'observations/slave_left_arm/joint',
                'observations/master_left_arm/joint'
            ]
            
            for path in left_arm_paths:
                try:
                    if path in f:
                        joint_data['left_arm'] = f[path][:]
                        if verbose:
                            print(f"Found left arm joints at '{path}': shape {joint_data['left_arm'].shape}")
                        break
                except:
                    continue
            
            # 如果上面的路径都没找到，尝试通过组结构查找
            if len(joint_data['left_arm']) == 0:
                left_arm_keys = ['left_arm', 'slave_left_arm', 'master_left_arm']
                for key in left_arm_keys:
                    if key in f:
                        left_arm_group = f[key]
                        if 'joint' in left_arm_group:
                            joint_data['left_arm'] = left_arm_group['joint'][:]
                            if verbose:
                                print(f"Found left arm joints in group '{key}': shape {joint_data['left_arm'].shape}")
                            break
                    # 也检查observations下的组
                    obs_key = f'observations/{key}'
                    if 'observations' in f and key in f['observations']:
                        obs_group = f['observations'][key]
                        if 'joint' in obs_group:
                            joint_data['left_arm'] = obs_group['joint'][:]
                            if verbose:
                                print(f"Found left arm joints in observations group '{key}': shape {joint_data['left_arm'].shape}")
                            break
            
            # Read right arm data (support multiple naming conventions and data structures)
            right_arm_paths = [
                # 直接路径格式
                'right_arm/joint',
                'slave_right_arm/joint',
                'master_right_arm/joint',
                # observations路径格式
                'observations/right_arm/joint',
                'observations/slave_right_arm/joint', 
                'observations/master_right_arm/joint'
            ]
            
            for path in right_arm_paths:
                try:
                    if path in f:
                        joint_data['right_arm'] = f[path][:]
                        if verbose:
                            print(f"Found right arm joints at '{path}': shape {joint_data['right_arm'].shape}")
                        break
                except:
                    continue
            
            # 如果上面的路径都没找到，尝试通过组结构查找
            if len(joint_data['right_arm']) == 0:
                right_arm_keys = ['right_arm', 'slave_right_arm', 'master_right_arm']
                for key in right_arm_keys:
                    if key in f:
                        right_arm_group = f[key]
                        if 'joint' in right_arm_group:
                            joint_data['right_arm'] = right_arm_group['joint'][:]
                            if verbose:
                                print(f"Found right arm joints in group '{key}': shape {joint_data['right_arm'].shape}")
                            break
                    # 也检查observations下的组
                    obs_key = f'observations/{key}'
                    if 'observations' in f and key in f['observations']:
                        obs_group = f['observations'][key]
                        if 'joint' in obs_group:
                            joint_data['right_arm'] = obs_group['joint'][:]
                            if verbose:
                                print(f"Found right arm joints in observations group '{key}': shape {joint_data['right_arm'].shape}")
                            break
                    
    except Exception as e:
        if verbose:
            print(f"Error reading {hdf5_path}: {str(e)}")
        return {'left_arm': [], 'right_arm': []}
    
    return joint_data


def collect_joint_data_from_folder(folder_path, label, verbose=False):
    """
    从文件夹中收集所有HDF5文件的joint数据
    
    Parameters:
        folder_path: 文件夹路径
        label: 数据源标签
        verbose: 是否输出详细信息
        
    Returns:
        dict: 收集到的所有joint数据
    """
    if not os.path.exists(folder_path):
        if verbose:
            print(f"文件夹不存在: {folder_path}")
        return {}
    
    # 查找所有HDF5文件
    hdf5_files = []
    for ext in ['*.hdf5', '*.h5']:
        hdf5_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not hdf5_files:
        if verbose:
            print(f"在文件夹 {folder_path} 中未找到HDF5文件")
        return {}
    
    if verbose:
        print(f"在 {folder_path} 中找到 {len(hdf5_files)} 个HDF5文件")
    
    collected_data = {
        'left_arm': [],
        'right_arm': [],
        'files': [],
        'label': label
    }
    
    # 处理每个HDF5文件
    for hdf5_file in tqdm(hdf5_files, desc=f"Reading {label} files", disable=not verbose):
        joint_data = read_joint_data_from_hdf5(hdf5_file, verbose=False)
        
        if len(joint_data['left_arm']) > 0:
            collected_data['left_arm'].append(joint_data['left_arm'])
            collected_data['files'].append(os.path.basename(hdf5_file))
        
        if len(joint_data['right_arm']) > 0:
            collected_data['right_arm'].append(joint_data['right_arm'])
    
    if verbose:
        print(f"{label} - 收集到左臂数据: {len(collected_data['left_arm'])} 个文件")
        print(f"{label} - 收集到右臂数据: {len(collected_data['right_arm'])} 个文件")
    
    return collected_data


def compare_joint_data(folder1_path, folder2_path, label1="Source 1", label2="Source 2", 
                      output_dir="output", verbose=False):
    """
    对比两个文件夹中HDF5文件的joint数据
    
    Parameters:
        folder1_path: 第一个文件夹路径
        folder2_path: 第二个文件夹路径
        label1: 第一个数据源的标签
        label2: 第二个数据源的标签
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"开始对比分析...")
        print(f"数据源1: {folder1_path} (标签: {label1})")
        print(f"数据源2: {folder2_path} (标签: {label2})")
        print(f"输出目录: {output_dir}")
        print("="*60)
    
    # 收集两个文件夹的数据
    data1 = collect_joint_data_from_folder(folder1_path, label1, verbose)
    data2 = collect_joint_data_from_folder(folder2_path, label2, verbose)
    
    if not data1 and not data2:
        print("两个文件夹都没有找到有效的HDF5文件")
        return
    
    # 对比左臂数据
    if data1.get('left_arm') or data2.get('right_arm'):
        visualize_joint_comparison(data1, data2, 'left_arm', output_dir, verbose)
    
    # 对比右臂数据
    if data1.get('right_arm') or data2.get('right_arm'):
        visualize_joint_comparison(data1, data2, 'right_arm', output_dir, verbose)
    
    if verbose:
        print(f"\n对比分析完成！结果保存在: {output_dir}")


def visualize_joint_comparison(data1, data2, arm_type, output_dir, verbose=False):
    """
    可视化特定手臂的joint数据对比
    使用seaborn绘图
    
    Parameters:
        data1: 第一个数据源的数据
        data2: 第二个数据源的数据
        arm_type: 'left_arm' 或 'right_arm'
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 获取数据
    joint_data1 = data1.get(arm_type, []) if data1 else []
    joint_data2 = data2.get(arm_type, []) if data2 else []
    
    if not joint_data1 and not joint_data2:
        if verbose:
            print(f"没有找到 {arm_type} 的数据")
        return
    
    # 确定关节数量
    max_joints = 0
    if joint_data1:
        max_joints = max(max_joints, max([data.shape[1] for data in joint_data1]))
    if joint_data2:
        max_joints = max(max_joints, max([data.shape[1] for data in joint_data2]))
    
    if max_joints == 0:
        if verbose:
            print(f"没有有效的 {arm_type} joint数据")
        return
    
    # 创建子图 - 每个关节一个子图
    cols = 3  # 每行3个子图
    rows = (max_joints + cols - 1) // cols  # 计算需要的行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    fig.suptitle(f'{arm_type.replace("_", " ").title()} Joint Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif max_joints == 1:
        axes = axes.reshape(-1, 1)
    
    # 为每个关节创建对比图
    for joint_idx in range(max_joints):
        row = joint_idx // cols
        col = joint_idx % cols
        ax = axes[row, col]
        
        # 绘制第一个数据源的数据
        if joint_data1:
            label1 = data1.get('label', 'Source 1')
            color1 = '#2E86C1'  # 蓝色
            
            for i, data in enumerate(joint_data1):
                if joint_idx < data.shape[1]:
                    time_steps = range(len(data))
                    if i == 0:  # 只在第一条线上加标签
                        sns.lineplot(x=time_steps, y=data[:, joint_idx], ax=ax,
                                   color=color1, alpha=0.7, linewidth=1.5, 
                                   label=f'{label1} (n={len(joint_data1)})')
                    else:
                        sns.lineplot(x=time_steps, y=data[:, joint_idx], ax=ax,
                                   color=color1, alpha=0.7, linewidth=1.5,
                                   legend=False)
        
        # 绘制第二个数据源的数据
        if joint_data2:
            label2 = data2.get('label', 'Source 2')
            color2 = '#E74C3C'  # 红色
            
            for i, data in enumerate(joint_data2):
                if joint_idx < data.shape[1]:
                    time_steps = range(len(data))
                    if i == 0:  # 只在第一条线上加标签
                        sns.lineplot(x=time_steps, y=data[:, joint_idx], ax=ax,
                                   color=color2, alpha=0.7, linewidth=1.5, 
                                   label=f'{label2} (n={len(joint_data2)})')
                    else:
                        sns.lineplot(x=time_steps, y=data[:, joint_idx], ax=ax,
                                   color=color2, alpha=0.7, linewidth=1.5,
                                   legend=False)
        
        # 设置子图属性
        ax.set_title(f'Joint {joint_idx + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Angle (rad)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 美化坐标轴
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 隐藏多余的子图
    for joint_idx in range(max_joints, rows * cols):
        row = joint_idx // cols
        col = joint_idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{arm_type}_joint_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存 {arm_type} 对比图: {output_path}")


def compare_joint_statistics(folder1_path, folder2_path, label1="Source 1", label2="Source 2", 
                           output_dir="output", verbose=False):
    """
    对比两个数据源的joint数据统计信息
    使用seaborn绘图
    
    Parameters:
        folder1_path: 第一个文件夹路径
        folder2_path: 第二个文件夹路径
        label1: 第一个数据源的标签
        label2: 第二个数据源的标签
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 收集数据
    data1 = collect_joint_data_from_folder(folder1_path, label1, verbose=False)
    data2 = collect_joint_data_from_folder(folder2_path, label2, verbose=False)
    
    # 创建统计图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Joint Data Statistics Comparison', fontsize=18, fontweight='bold')
    
    # 统计每个关节的平均值和标准差
    arms = ['left_arm', 'right_arm']
    colors = ['#2E86C1', '#E74C3C']  # 蓝色和红色
    labels = [label1, label2]
    
    for arm_idx, arm_type in enumerate(arms):
        ax_mean = ax1 if arm_idx == 0 else ax2
        ax_std = ax3 if arm_idx == 0 else ax4
        
        # 准备数据用于seaborn绘图
        mean_data = []
        std_data = []
        
        # 计算统计信息
        for data_idx, (data, color, label) in enumerate(zip([data1, data2], colors, labels)):
            joint_data = data.get(arm_type, [])
            if not joint_data:
                continue
                
            # 合并所有数据
            all_data = np.concatenate(joint_data, axis=0)
            
            # 计算每个关节的统计信息
            joint_means = np.mean(all_data, axis=0)
            joint_stds = np.std(all_data, axis=0)
            
            # 为seaborn准备数据
            for joint_idx, (mean_val, std_val) in enumerate(zip(joint_means, joint_stds)):
                mean_data.append({
                    'Joint': f'Joint {joint_idx + 1}',
                    'Value': mean_val,
                    'Source': f'{label} (n={len(joint_data)})',
                    'Color': color
                })
                std_data.append({
                    'Joint': f'Joint {joint_idx + 1}',
                    'Value': std_val,
                    'Source': f'{label} (n={len(joint_data)})',
                    'Color': color
                })
        
        # 使用seaborn绘制柱状图
        if mean_data:
            mean_df = pd.DataFrame(mean_data)
            sns.barplot(data=mean_df, x='Joint', y='Value', hue='Source', 
                       ax=ax_mean, alpha=0.8)
        
        if std_data:
            std_df = pd.DataFrame(std_data)
            sns.barplot(data=std_df, x='Joint', y='Value', hue='Source', 
                       ax=ax_std, alpha=0.8)
        
        # 设置图表属性
        ax_mean.set_title(f'{arm_type.replace("_", " ").title()} - Joint Mean Values', 
                         fontsize=14, fontweight='bold')
        ax_mean.set_xlabel('Joint Index', fontsize=12)
        ax_mean.set_ylabel('Mean Angle (rad)', fontsize=12)
        ax_mean.legend(fontsize=10)
        ax_mean.grid(True, alpha=0.3)
        ax_mean.tick_params(labelsize=10)
        
        # 美化坐标轴
        ax_mean.spines['top'].set_visible(False)
        ax_mean.spines['right'].set_visible(False)
        
        ax_std.set_title(f'{arm_type.replace("_", " ").title()} - Joint Standard Deviation', 
                        fontsize=14, fontweight='bold')
        ax_std.set_xlabel('Joint Index', fontsize=12)
        ax_std.set_ylabel('Std Angle (rad)', fontsize=12)
        ax_std.legend(fontsize=10)
        ax_std.grid(True, alpha=0.3)
        ax_std.tick_params(labelsize=10)
        
        # 美化坐标轴
        ax_std.spines['top'].set_visible(False)
        ax_std.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存统计图
    stats_path = os.path.join(output_dir, 'joint_statistics_comparison.png')
    plt.savefig(stats_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存统计对比图: {stats_path}")


def plot_average_joint_curves(folder1_path, folder2_path, label1="Source 1", label2="Source 2", 
                            output_dir="output", verbose=False):
    """
    绘制源数据和测试数据的同一时刻平均角度曲线
    使用seaborn绘图，线条加粗，单独保存图片
    
    Parameters:
        folder1_path: 第一个文件夹路径
        folder2_path: 第二个文件夹路径
        label1: 第一个数据源的标签
        label2: 第二个数据源的标签
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 收集数据
    data1 = collect_joint_data_from_folder(folder1_path, label1, verbose=False)
    data2 = collect_joint_data_from_folder(folder2_path, label2, verbose=False)
    
    if not data1 and not data2:
        if verbose:
            print("没有找到有效数据，无法绘制平均曲线")
        return
    
    # 为每个手臂绘制平均曲线
    arms = ['left_arm', 'right_arm']
    
    for arm_type in arms:
        joint_data1 = data1.get(arm_type, []) if data1 else []
        joint_data2 = data2.get(arm_type, []) if data2 else []
        
        if not joint_data1 and not joint_data2:
            if verbose:
                print(f"没有找到 {arm_type} 的数据")
            continue
        
        # 计算平均曲线
        avg_curves1 = calculate_average_curves(joint_data1, verbose)
        avg_curves2 = calculate_average_curves(joint_data2, verbose)
        
        # 确定最大关节数和最大时间步长
        max_joints = 0
        max_timesteps = 0
        
        if avg_curves1 is not None:
            max_joints = max(max_joints, avg_curves1.shape[1])
            max_timesteps = max(max_timesteps, avg_curves1.shape[0])
        if avg_curves2 is not None:
            max_joints = max(max_joints, avg_curves2.shape[1])
            max_timesteps = max(max_timesteps, avg_curves2.shape[0])
        
        if max_joints == 0:
            continue
        
        # 创建图形
        cols = 3
        rows = (max_joints + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
        fig.suptitle(f'{arm_type.replace("_", " ").title()} - Average Joint Angle Curves', 
                     fontsize=18, fontweight='bold')
        
        # 确保axes是二维数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif max_joints == 1:
            axes = axes.reshape(-1, 1)
        
        # 为每个关节绘制平均曲线
        for joint_idx in range(max_joints):
            row = joint_idx // cols
            col = joint_idx % cols
            ax = axes[row, col]
            
            # 绘制第一个数据源的平均曲线
            if avg_curves1 is not None and joint_idx < avg_curves1.shape[1]:
                time_steps = range(len(avg_curves1))
                sns.lineplot(x=time_steps, y=avg_curves1[:, joint_idx], 
                           ax=ax, label=f'{label1} (Avg)', 
                           linewidth=3, color='#2E86C1')  # 蓝色，加粗
            
            # 绘制第二个数据源的平均曲线
            if avg_curves2 is not None and joint_idx < avg_curves2.shape[1]:
                time_steps = range(len(avg_curves2))
                sns.lineplot(x=time_steps, y=avg_curves2[:, joint_idx], 
                           ax=ax, label=f'{label2} (Avg)', 
                           linewidth=3, color='#E74C3C')  # 红色，加粗
            
            # 设置子图属性
            ax.set_title(f'Joint {joint_idx + 1}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Angle (rad)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 美化坐标轴
            ax.tick_params(labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 隐藏多余的子图
        for joint_idx in range(max_joints, rows * cols):
            row = joint_idx // cols
            col = joint_idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 保存图像
        output_path = os.path.join(output_dir, f'{arm_type}_average_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if verbose:
            print(f"保存 {arm_type} 平均曲线图: {output_path}")


def calculate_average_curves(joint_data_list, verbose=False):
    """
    计算多个joint数据的平均曲线
    
    Parameters:
        joint_data_list: joint数据列表
        verbose: 是否输出详细信息
        
    Returns:
        numpy.ndarray: 平均曲线数据，形状为 (timesteps, joints)
    """
    if not joint_data_list:
        return None
    
    # 找到最小的时间步长和关节数，以便对齐数据
    min_timesteps = min([data.shape[0] for data in joint_data_list])
    min_joints = min([data.shape[1] for data in joint_data_list])
    
    if verbose:
        print(f"计算平均曲线 - 最小时间步: {min_timesteps}, 最小关节数: {min_joints}")
    
    # 截取所有数据到相同的维度
    aligned_data = []
    for data in joint_data_list:
        aligned_data.append(data[:min_timesteps, :min_joints])
    
    # 计算平均值
    stacked_data = np.stack(aligned_data, axis=0)  # shape: (n_files, timesteps, joints)
    average_curves = np.mean(stacked_data, axis=0)  # shape: (timesteps, joints)
    
    if verbose:
        print(f"平均曲线形状: {average_curves.shape}")
    
    return average_curves


if __name__ == "__main__":
    # 示例用法
    folder1 = "/home/usst/kwj/GitCode/control_your_robot_jie/save/real_data/stack_bowls_two_zip/"
    folder2 = "/home/usst/kwj/GitCode/control_your_robot_jie/test/reload_model_actions/stack_bowls_two/"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder1):
        print(f"文件夹1不存在: {folder1}")
        print("请修改folder1路径")
    elif not os.path.exists(folder2):
        print(f"文件夹2不存在: {folder2}")
        print("请修改folder2路径")
    else:
        # 执行对比分析
        output_dir = "save/output/joint_comparison"
        compare_joint_data(
            folder1_path=folder1,
            folder2_path=folder2,
            label1="ACT TestData",
            label2="Original Data",
            output_dir=output_dir,
            verbose=True
        )
        
        # 生成统计对比
        compare_joint_statistics(
            folder1_path=folder1,
            folder2_path=folder2,
            label1="ACT Data",
            label2="Original Data",
            output_dir=output_dir,
            verbose=True
        )
        
        # 绘制平均角度曲线
        plot_average_joint_curves(
            folder1_path=folder1,
            folder2_path=folder2,
            label1="ACT Data",
            label2="Original Data",
            output_dir=output_dir,
            verbose=True
        )
