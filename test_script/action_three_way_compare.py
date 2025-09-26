#!/usr/bin/env python3
"""
对比三个文件夹下HDF5文件中的action数据
- 源数据文件夹：使用 slave_left_arm 键名
- 模型1文件夹：使用 left_arm 键名  
- 模型2文件夹：使用 left_arm 键名
支持自动匹配测试数据中包含的HDF5文件，生成完整的对比分析图表

使用方法:
    conda activate telerobot
    python action_three_way_compare.py --source_folder /path/to/source --model1_folder /path/to/model1 --model2_folder /path/to/model2
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from tqdm import tqdm
import glob
import sys


def read_action_data_from_hdf5(hdf5_path, key_name, verbose=False):
    """
    从HDF5文件中读取指定键名的action数据
    
    Parameters:
        hdf5_path: HDF5文件路径
        key_name: 键名 ('slave_left_arm' 或 'left_arm')
        verbose: 是否输出详细信息
        
    Returns:
        numpy.ndarray: action数据，如果没找到则返回None
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if verbose:
                print(f"文件 {os.path.basename(hdf5_path)} 包含的组:")
                def print_structure(name, obj):
                    print(f"  {name}: {type(obj)}")
                f.visititems(print_structure)
            
            # 根据键名尝试不同的路径
            if key_name == 'slave_left_arm':
                action_paths = [
                    'slave_left_arm/joint',
                    'slave_left_arm/qpos',
                    'slave_left_arm',
                    'observations/slave_left_arm/joint',
                    'observations/slave_left_arm/qpos'
                ]
            elif key_name == 'left_arm':
                action_paths = [
                    'left_arm/joint',
                    'left_arm/qpos',
                    'left_arm',
                    'action/left_arm',
                    'observations/left_arm/joint',
                    'observations/left_arm/qpos',
                    'action/model_output',  # 模型输出格式
                    'action',               # 直接的action数据集
                    'actions'               # 复数形式
                ]
            else:
                action_paths = [key_name]
            
            for path in action_paths:
                try:
                    if path in f:
                        action_data = f[path][:]
                        if verbose:
                            print(f"在路径 '{path}' 找到action数据: shape {action_data.shape}")
                        
                        # 处理维度问题：如果是3维数据，转换为2维
                        if len(action_data.shape) == 3 and action_data.shape[1] == 1:
                            action_data = action_data.squeeze(axis=1)
                        
                        return action_data
                        
                    # 处理嵌套路径
                    elif '/' in path:
                        parts = path.split('/')
                        current = f
                        found = True
                        for part in parts:
                            if part in current:
                                current = current[part]
                            else:
                                found = False
                                break
                        if found and hasattr(current, 'shape'):
                            action_data = current[:]
                            if verbose:
                                print(f"在路径 '{path}' 找到action数据: shape {action_data.shape}")
                            
                            # 处理维度问题
                            if len(action_data.shape) == 3 and action_data.shape[1] == 1:
                                action_data = action_data.squeeze(axis=1)
                            
                            return action_data
                except Exception as e:
                    continue
                    
    except Exception as e:
        if verbose:
            print(f"读取文件 {hdf5_path} 时出错: {str(e)}")
        return None
    
    if verbose:
        print(f"在文件 {hdf5_path} 中未找到键名 '{key_name}' 的数据")
    
    return None


def collect_action_data_from_folder(folder_path, key_name, label, verbose=False, file_filter=None):
    """
    从文件夹中收集所有HDF5文件的action数据
    
    Parameters:
        folder_path: 文件夹路径
        key_name: HDF5文件中的键名
        label: 数据源标签
        verbose: 是否输出详细信息
        file_filter: 可选的文件名过滤列表，只处理这些文件
        
    Returns:
        dict: 收集到的所有action数据
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
    
    # 如果提供了文件过滤器，只处理指定的文件
    if file_filter is not None:
        filtered_files = []
        for hdf5_file in hdf5_files:
            filename = os.path.basename(hdf5_file)
            if filename in file_filter:
                filtered_files.append(hdf5_file)
        hdf5_files = filtered_files
    
    if verbose:
        print(f"在 {folder_path} 中找到 {len(hdf5_files)} 个HDF5文件")
        print(f"使用键名: {key_name}")
    
    collected_data = {
        'actions': [],
        'files': [],
        'label': label
    }
    
    # 按文件名排序，确保处理顺序一致
    hdf5_files.sort()
    
    # 处理每个HDF5文件
    success_count = 0
    for hdf5_file in tqdm(hdf5_files, desc=f"Reading {label} files", disable=not verbose):
        action_data = read_action_data_from_hdf5(hdf5_file, key_name, verbose=False)
        
        if action_data is not None:
            collected_data['actions'].append(action_data)
            collected_data['files'].append(os.path.basename(hdf5_file))
            success_count += 1
    
    if verbose:
        print(f"{label} - 成功读取: {success_count}/{len(hdf5_files)} 个文件")
        if collected_data['actions']:
            shapes = [data.shape for data in collected_data['actions']]
            print(f"数据形状: {shapes[:5]}...")  # 只显示前5个
    
    return collected_data


def find_common_files_three_way(source_folder, model1_folder, model2_folder, verbose=False):
    """
    找到三个文件夹中都存在的HDF5文件
    支持多种命名格式的自动识别和匹配
    
    Parameters:
        source_folder: 源数据文件夹路径
        model1_folder: 模型1文件夹路径
        model2_folder: 模型2文件夹路径
        verbose: 是否输出详细信息
        
    Returns:
        list: 匹配的文件三元组列表，每个元素为 (source_file, model1_file, model2_file)
    """
    if not all([os.path.exists(source_folder), os.path.exists(model1_folder), os.path.exists(model2_folder)]):
        if verbose:
            print("一个或多个文件夹不存在")
        return []
    
    def extract_number_from_filename(filename):
        """从文件名中提取数字，支持多种格式"""
        name_without_ext = os.path.splitext(filename)[0]
        
        # 格式1: 纯数字 (0.hdf5, 1.hdf5)
        if name_without_ext.isdigit():
            return int(name_without_ext)
        
        # 格式2: episode_数字 (episode_0.hdf5, episode_1.hdf5)
        if name_without_ext.startswith('episode_'):
            number_part = name_without_ext.replace('episode_', '')
            if number_part.isdigit():
                return int(number_part)
        
        # 格式3: 其他可能的前缀_数字格式
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                return int(parts[-1])
        
        return None
    
    # 获取三个文件夹中的所有HDF5文件
    folders = [source_folder, model1_folder, model2_folder]
    folder_files = []  # [{number: filename}, {number: filename}, {number: filename}]
    
    for folder_path in folders:
        files_dict = {}
        for ext in ['*.hdf5', '*.h5']:
            for file_path in glob.glob(os.path.join(folder_path, ext)):
                filename = os.path.basename(file_path)
                number = extract_number_from_filename(filename)
                if number is not None:
                    files_dict[number] = filename
        folder_files.append(files_dict)
    
    # 找到三个文件夹中都存在的数字（任务编号）
    common_numbers = set(folder_files[0].keys())
    for files_dict in folder_files[1:]:
        common_numbers = common_numbers.intersection(set(files_dict.keys()))
    
    common_numbers = sorted(list(common_numbers))
    
    # 创建匹配的文件三元组列表
    common_file_triplets = []
    for number in common_numbers:
        source_file = folder_files[0][number]
        model1_file = folder_files[1][number]
        model2_file = folder_files[2][number]
        common_file_triplets.append((source_file, model1_file, model2_file))
    
    if verbose:
        folder_names = ['源数据', '模型1', '模型2']
        for i, (folder_name, files_dict) in enumerate(zip(folder_names, folder_files)):
            print(f"{folder_name}文件夹 ({folders[i]}):")
            print(f"  - 文件数: {len(files_dict)}")
            print(f"  - 任务编号: {sorted(files_dict.keys())}")
        
        print(f"三方匹配结果:")
        print(f"  - 匹配的任务数: {len(common_file_triplets)}")
        if common_file_triplets:
            print(f"  - 匹配的文件三元组:")
            for source, model1, model2 in common_file_triplets[:5]:  # 只显示前5个
                print(f"    {source} <-> {model1} <-> {model2}")
            if len(common_file_triplets) > 5:
                print(f"    ... 还有 {len(common_file_triplets) - 5} 个文件三元组")
    
    return common_file_triplets


def visualize_action_comparison_three_way(data_source, data_model1, data_model2, output_dir, verbose=False):
    """
    可视化三方action数据对比 - 所有轨迹叠加显示
    使用seaborn绘图，标签统一放在右上角
    
    Parameters:
        data_source: 源数据
        data_model1: 模型1数据
        data_model2: 模型2数据
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式和matplotlib参数
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    
    # 获取数据
    actions_source = data_source.get('actions', [])
    actions_model1 = data_model1.get('actions', [])
    actions_model2 = data_model2.get('actions', [])
    
    # 检查是否有有效数据
    valid_data = []
    if actions_source:
        valid_data.append(actions_source)
    if actions_model1:
        valid_data.append(actions_model1)
    if actions_model2:
        valid_data.append(actions_model2)
    
    if not valid_data:
        if verbose:
            print("没有找到action数据")
        return
    
    # 确定共同的维度数
    all_dims = []
    for actions in valid_data:
        if actions:
            all_dims.append(min([data.shape[1] for data in actions]))
    
    common_dims = min(all_dims) if all_dims else 0
    
    if common_dims == 0:
        if verbose:
            print("没有有效的action数据或没有共同的维度")
        return
    
    if verbose:
        print(f"将对比共同的 {common_dims} 个维度")
    
    # 创建子图 - 每个action维度一个子图
    cols = 3  # 每行3个子图
    rows = (common_dims + cols - 1) // cols  # 计算需要的行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    
    # 创建更详细的标题，显示文件数量信息
    n_source = len(actions_source) if actions_source else 0
    n_model1 = len(actions_model1) if actions_model1 else 0
    n_model2 = len(actions_model2) if actions_model2 else 0
    title = f'Action Data Comparison - All Trajectories\n'
    title += f'Source: {n_source} files | Model1: {n_model1} files | Model2: {n_model2} files'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    
    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif common_dims == 1:
        axes = axes.reshape(-1, 1)
    
    # 定义颜色方案
    colors = {
        'source': '#2E86C1',    # 蓝色 - 源数据
        'model1': '#E74C3C',    # 红色 - 模型1
        'model2': '#28B463'     # 绿色 - 模型2
    }
    
    # 为每个action维度创建对比图
    for dim_idx in range(common_dims):
        row = dim_idx // cols
        col = dim_idx % cols
        ax = axes[row, col]
        
        # 绘制源数据
        if actions_source:
            label_source = data_source.get('label', 'Source')
            
            for i, data in enumerate(actions_source):
                if dim_idx < data.shape[1]:
                    time_steps = range(len(data))
                    if i == 0:  # 只在第一条线上加标签
                        sns.lineplot(x=time_steps, y=data[:, dim_idx], ax=ax,
                                   color=colors['source'], alpha=0.7, linewidth=1.5, 
                                   label=f'{label_source} (n={len(actions_source)})')
                    else:
                        sns.lineplot(x=time_steps, y=data[:, dim_idx], ax=ax,
                                   color=colors['source'], alpha=0.7, linewidth=1.5,
                                   legend=False)
        
        # 绘制模型1数据
        if actions_model1:
            label_model1 = data_model1.get('label', 'Model1')
            
            for i, data in enumerate(actions_model1):
                if dim_idx < data.shape[1]:
                    time_steps = range(len(data))
                    if i == 0:  # 只在第一条线上加标签
                        sns.lineplot(x=time_steps, y=data[:, dim_idx], ax=ax,
                                   color=colors['model1'], alpha=0.7, linewidth=1.5, 
                                   label=f'{label_model1} (n={len(actions_model1)})')
                    else:
                        sns.lineplot(x=time_steps, y=data[:, dim_idx], ax=ax,
                                   color=colors['model1'], alpha=0.7, linewidth=1.5,
                                   legend=False)
        
        # 绘制模型2数据
        if actions_model2:
            label_model2 = data_model2.get('label', 'Model2')
            
            for i, data in enumerate(actions_model2):
                if dim_idx < data.shape[1]:
                    time_steps = range(len(data))
                    if i == 0:  # 只在第一条线上加标签
                        sns.lineplot(x=time_steps, y=data[:, dim_idx], ax=ax,
                                   color=colors['model2'], alpha=0.7, linewidth=1.5, 
                                   label=f'{label_model2} (n={len(actions_model2)})')
                    else:
                        sns.lineplot(x=time_steps, y=data[:, dim_idx], ax=ax,
                                   color=colors['model2'], alpha=0.7, linewidth=1.5,
                                   legend=False)
        
        # 设置子图属性
        ax.set_title(f'Dimension {dim_idx + 1}', fontweight='bold', pad=15)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        
        # 将图例放在右上角，避免遮盖数据
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                 framealpha=0.9, borderpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
    
    # 隐藏多余的子图
    for dim_idx in range(common_dims, rows * cols):
        row = dim_idx // cols
        col = dim_idx % cols
        axes[row, col].set_visible(False)
    
    # 调整布局，避免重叠
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 保存图像
    output_path = os.path.join(output_dir, 'three_way_action_comparison_all_trajectories.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存三方action对比图: {output_path}")


def compare_action_statistics_three_way(data_source, data_model1, data_model2, output_dir, verbose=False):
    """
    对比三个数据源的action数据统计信息
    使用seaborn绘图，标签统一放在右上角
    
    Parameters:
        data_source: 源数据
        data_model1: 模型1数据
        data_model2: 模型2数据
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    
    # 创建统计图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 获取数据源信息
    label_source = data_source.get('label', 'Source')
    label_model1 = data_model1.get('label', 'Model1')
    label_model2 = data_model2.get('label', 'Model2')
    n_source = len(data_source.get('actions', []))
    n_model1 = len(data_model1.get('actions', []))
    n_model2 = len(data_model2.get('actions', []))
    
    title = f'Action Data Statistics Comparison\n'
    title += f'Source: {n_source} files | Model1: {n_model1} files | Model2: {n_model2} files'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    
    # 确定共同的维度数
    all_actions = [
        data_source.get('actions', []),
        data_model1.get('actions', []),
        data_model2.get('actions', [])
    ]
    
    all_dims = []
    for actions in all_actions:
        if actions:
            all_dims.append(min([data.shape[1] for data in actions]))
    
    common_dims = min(all_dims) if all_dims else 0
    
    if common_dims == 0:
        if verbose:
            print("没有有效的action数据或没有共同的维度")
        return
    
    if verbose:
        print(f"统计对比 - 将对比共同的 {common_dims} 个维度")
    
    # 准备数据用于seaborn绘图
    mean_data = []
    std_data = []
    range_data = []
    var_data = []
    
    colors = ['#2E86C1', '#E74C3C', '#28B463']  # 蓝色、红色、绿色
    data_list = [data_source, data_model1, data_model2]
    
    # 计算统计信息
    for data_idx, (data, color) in enumerate(zip(data_list, colors)):
        actions = data.get('actions', [])
        label = data.get('label', f'Source {data_idx + 1}')
        
        if not actions:
            continue
            
        # 合并所有action数据，只取共同的维度
        all_actions = np.concatenate([action[:, :common_dims] for action in actions], axis=0)
        
        # 计算每个维度的统计信息
        action_means = np.mean(all_actions, axis=0)
        action_stds = np.std(all_actions, axis=0)
        action_ranges = np.ptp(all_actions, axis=0)  # peak-to-peak (max - min)
        action_vars = np.var(all_actions, axis=0)
        
        # 为seaborn准备数据，只处理共同的维度
        for dim_idx, (mean_val, std_val, range_val, var_val) in enumerate(
            zip(action_means, action_stds, action_ranges, action_vars)):
            source_label = f'{label} (n={len(actions)})'
            
            mean_data.append({
                'Dimension': f'Dim {dim_idx + 1}',
                'Value': mean_val,
                'Source': source_label
            })
            std_data.append({
                'Dimension': f'Dim {dim_idx + 1}',
                'Value': std_val,
                'Source': source_label
            })
            range_data.append({
                'Dimension': f'Dim {dim_idx + 1}',
                'Value': range_val,
                'Source': source_label
            })
            var_data.append({
                'Dimension': f'Dim {dim_idx + 1}',
                'Value': var_val,
                'Source': source_label
            })
    
    # 使用seaborn绘制柱状图
    if mean_data:
        mean_df = pd.DataFrame(mean_data)
        sns.barplot(data=mean_df, x='Dimension', y='Value', hue='Source', 
                   ax=ax1, alpha=0.8)
        ax1.set_title('Action Mean Values', fontweight='bold')
        ax1.set_ylabel('Mean Value')
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    if std_data:
        std_df = pd.DataFrame(std_data)
        sns.barplot(data=std_df, x='Dimension', y='Value', hue='Source', 
                   ax=ax2, alpha=0.8)
        ax2.set_title('Action Standard Deviation', fontweight='bold')
        ax2.set_ylabel('Std Value')
        ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    if range_data:
        range_df = pd.DataFrame(range_data)
        sns.barplot(data=range_df, x='Dimension', y='Value', hue='Source', 
                   ax=ax3, alpha=0.8)
        ax3.set_title('Action Range (Max - Min)', fontweight='bold')
        ax3.set_ylabel('Range Value')
        ax3.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    if var_data:
        var_df = pd.DataFrame(var_data)
        sns.barplot(data=var_df, x='Dimension', y='Value', hue='Source', 
                   ax=ax4, alpha=0.8)
        ax4.set_title('Action Variance', fontweight='bold')
        ax4.set_ylabel('Variance Value')
        ax4.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # 美化所有子图
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 保存统计图
    stats_path = os.path.join(output_dir, 'three_way_action_statistics_comparison.png')
    plt.savefig(stats_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存三方统计对比图: {stats_path}")


def calculate_average_actions(action_list, verbose=False, max_dims=None):
    """
    计算多个action数据的平均曲线
    
    Parameters:
        action_list: action数据列表
        verbose: 是否输出详细信息
        max_dims: 可选，指定最大维度数
        
    Returns:
        numpy.ndarray: 平均曲线数据，形状为 (timesteps, action_dims)
    """
    if not action_list:
        return None
    
    # 找到最小的时间步长和action维度，以便对齐数据
    min_timesteps = min([data.shape[0] for data in action_list])
    min_dims = min([data.shape[1] for data in action_list])
    
    # 如果指定了max_dims，使用较小的值
    if max_dims is not None:
        min_dims = min(min_dims, max_dims)
    
    if verbose:
        timesteps_info = [data.shape[0] for data in action_list]
        dims_info = [data.shape[1] for data in action_list]
        print(f"时间步长范围: {min(timesteps_info)} - {max(timesteps_info)}, 使用: {min_timesteps}")
        print(f"维度范围: {min(dims_info)} - {max(dims_info)}, 使用: {min_dims}")
    
    # 截取所有数据到相同的维度
    aligned_data = []
    for data in action_list:
        aligned_data.append(data[:min_timesteps, :min_dims])
    
    # 计算平均值
    stacked_data = np.stack(aligned_data, axis=0)  # shape: (n_files, timesteps, dims)
    average_actions = np.mean(stacked_data, axis=0)  # shape: (timesteps, dims)
    
    if verbose:
        print(f"平均action曲线形状: {average_actions.shape}")
    
    return average_actions


def plot_action_overlay_three_way(data_source, data_model1, data_model2, output_dir, verbose=False):
    """
    绘制三方action数据的平均曲线叠加图
    使用seaborn绘图，标签统一放在右上角
    
    Parameters:
        data_source: 源数据
        data_model1: 模型1数据
        data_model2: 模型2数据
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    
    # 获取数据
    actions_source = data_source.get('actions', [])
    actions_model1 = data_model1.get('actions', [])
    actions_model2 = data_model2.get('actions', [])
    
    # 检查是否有有效数据
    valid_data = []
    if actions_source:
        valid_data.append(actions_source)
    if actions_model1:
        valid_data.append(actions_model1)
    if actions_model2:
        valid_data.append(actions_model2)
    
    if not valid_data:
        if verbose:
            print("没有找到action数据，无法绘制平均曲线")
        return
    
    # 确定共同的维度数
    all_dims = []
    for actions in valid_data:
        if actions:
            all_dims.append(min([data.shape[1] for data in actions]))
    
    common_dims = min(all_dims) if all_dims else 0
    
    if common_dims == 0:
        if verbose:
            print("没有有效的action数据或没有共同的维度")
        return
    
    if verbose:
        print(f"平均曲线对比 - 将对比共同的 {common_dims} 个维度")
    
    # 计算平均曲线，只使用共同的维度
    avg_actions_source = calculate_average_actions(actions_source, verbose, max_dims=common_dims)
    avg_actions_model1 = calculate_average_actions(actions_model1, verbose, max_dims=common_dims)
    avg_actions_model2 = calculate_average_actions(actions_model2, verbose, max_dims=common_dims)
    
    # 创建图形
    cols = 3
    rows = (common_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    
    # 创建更详细的标题
    label_source = data_source.get('label', 'Source')
    label_model1 = data_model1.get('label', 'Model1')
    label_model2 = data_model2.get('label', 'Model2')
    n_source = len(actions_source) if actions_source else 0
    n_model1 = len(actions_model1) if actions_model1 else 0
    n_model2 = len(actions_model2) if actions_model2 else 0
    
    title = f'Action Data - Average Curves Comparison\n'
    title += f'Source: {n_source} files | Model1: {n_model1} files | Model2: {n_model2} files'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    
    # 确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif common_dims == 1:
        axes = axes.reshape(-1, 1)
    
    # 定义颜色方案
    colors = {
        'source': '#2E86C1',    # 蓝色 - 源数据
        'model1': '#E74C3C',    # 红色 - 模型1
        'model2': '#28B463'     # 绿色 - 模型2
    }
    
    # 为每个维度绘制平均曲线
    for dim_idx in range(common_dims):
        row = dim_idx // cols
        col = dim_idx % cols
        ax = axes[row, col]
        
        # 绘制源数据的平均曲线
        if avg_actions_source is not None and dim_idx < avg_actions_source.shape[1]:
            time_steps = range(len(avg_actions_source))
            sns.lineplot(x=time_steps, y=avg_actions_source[:, dim_idx], 
                       ax=ax, label=f'{label_source} (Avg)', 
                       linewidth=3, color=colors['source'])
        
        # 绘制模型1的平均曲线
        if avg_actions_model1 is not None and dim_idx < avg_actions_model1.shape[1]:
            time_steps = range(len(avg_actions_model1))
            sns.lineplot(x=time_steps, y=avg_actions_model1[:, dim_idx], 
                       ax=ax, label=f'{label_model1} (Avg)', 
                       linewidth=3, color=colors['model1'])
        
        # 绘制模型2的平均曲线
        if avg_actions_model2 is not None and dim_idx < avg_actions_model2.shape[1]:
            time_steps = range(len(avg_actions_model2))
            sns.lineplot(x=time_steps, y=avg_actions_model2[:, dim_idx], 
                       ax=ax, label=f'{label_model2} (Avg)', 
                       linewidth=3, color=colors['model2'])
        
        # 设置子图属性
        ax.set_title(f'Dimension {dim_idx + 1}', fontweight='bold', pad=15)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        
        # 将图例放在右上角，避免遮盖数据
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                 framealpha=0.9, borderpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
    
    # 隐藏多余的子图
    for dim_idx in range(common_dims, rows * cols):
        row = dim_idx // cols
        col = dim_idx % cols
        axes[row, col].set_visible(False)
    
    # 调整布局，避免重叠
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 保存图像
    output_path = os.path.join(output_dir, 'three_way_action_average_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存三方action平均曲线图: {output_path}")


def compare_three_folders(source_folder, model1_folder, model2_folder, 
                         label_source="Source Data", label_model1="Model 1", label_model2="Model 2",
                         output_dir="output", verbose=False):
    """
    对比三个文件夹中所有HDF5文件的action数据
    支持多种命名格式的自动匹配
    只对比三个文件夹中都存在对应任务编号的文件
    
    Parameters:
        source_folder: 源数据文件夹路径（使用slave_left_arm键名）
        model1_folder: 模型1文件夹路径（使用left_arm键名）
        model2_folder: 模型2文件夹路径（使用left_arm键名）
        label_source: 源数据的标签
        label_model1: 模型1的标签
        label_model2: 模型2的标签
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("="*80)
        print("开始对比三个文件夹的action数据...")
        print(f"源数据文件夹: {source_folder} (键名: slave_left_arm)")
        print(f"模型1文件夹: {model1_folder} (键名: left_arm)")
        print(f"模型2文件夹: {model2_folder} (键名: left_arm)")
        print(f"输出目录: {output_dir}")
        print("="*80)
    
    # 找到三个文件夹中匹配的文件三元组
    common_file_triplets = find_common_files_three_way(source_folder, model1_folder, model2_folder, verbose)
    
    if not common_file_triplets:
        print("三个文件夹中没有找到匹配的HDF5文件")
        return
    
    if verbose:
        print(f"\n将对比 {len(common_file_triplets)} 组匹配的文件")
        print("="*80)
    
    # 分别收集三个文件夹的数据
    source_files_filter = [triplet[0] for triplet in common_file_triplets]  # 源数据文件
    model1_files_filter = [triplet[1] for triplet in common_file_triplets]  # 模型1文件
    model2_files_filter = [triplet[2] for triplet in common_file_triplets]  # 模型2文件
    
    data_source = collect_action_data_from_folder(source_folder, 'slave_left_arm', label_source, verbose, file_filter=source_files_filter)
    data_model1 = collect_action_data_from_folder(model1_folder, 'left_arm', label_model1, verbose, file_filter=model1_files_filter)
    data_model2 = collect_action_data_from_folder(model2_folder, 'left_arm', label_model2, verbose, file_filter=model2_files_filter)
    
    if not data_source.get('actions') and not data_model1.get('actions') and not data_model2.get('actions'):
        print("在匹配的文件中没有找到有效的action数据")
        return
    
    # 生成对比图
    visualize_action_comparison_three_way(data_source, data_model1, data_model2, output_dir, verbose)
    compare_action_statistics_three_way(data_source, data_model1, data_model2, output_dir, verbose)
    plot_action_overlay_three_way(data_source, data_model1, data_model2, output_dir, verbose)
    
    if verbose:
        print("\n" + "="*80)
        print("三方对比分析完成！")
        print(f"结果保存在: {output_dir}")
        print(f"源数据 ({label_source}): {len(data_source.get('actions', []))} 个文件")
        print(f"模型1 ({label_model1}): {len(data_model1.get('actions', []))} 个文件")
        print(f"模型2 ({label_model2}): {len(data_model2.get('actions', []))} 个文件")
        print("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="对比三个文件夹下HDF5文件中的action数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    conda activate telerobot
    python action_three_way_compare.py --source_folder /path/to/source --model1_folder /path/to/model1 --model2_folder /path/to/model2
    python action_three_way_compare.py --source_folder save/pick_place_cup/ --model1_folder test/act_pick_place_cup/ --model2_folder test/pi0_pick_place_cup/ --verbose
        """
    )
    
    parser.add_argument("--source_folder", type=str, required=True,
                       help="源数据文件夹路径 (使用slave_left_arm键名)")
    parser.add_argument("--model1_folder", type=str, required=True,
                       help="模型1文件夹路径 (使用left_arm键名)")
    parser.add_argument("--model2_folder", type=str, required=True,
                       help="模型2文件夹路径 (使用left_arm键名)")
    parser.add_argument("--label_source", type=str, default="Source Data",
                       help="源数据的标签 (默认: Source Data)")
    parser.add_argument("--label_model1", type=str, default="Model 1",
                       help="模型1的标签 (默认: Model 1)")
    parser.add_argument("--label_model2", type=str, default="Model 2",
                       help="模型2的标签 (默认: Model 2)")
    parser.add_argument("--output", type=str, default="save/output/three_way_action_comparison",
                       help="输出目录 (默认: save/output/three_way_action_comparison)")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细信息")
    
    return parser.parse_args()


if __name__ == "__main__":
    '''
    使用示例
    python test_script/action_three_way_compare.py 
    --source_folder ./save/pick_place_cup/ 
    --model1_folder ./test/reload_model_actions/act_pick_place_cup/ 
    --model2_folder ./test/reload_model_actions/pi0_pick_place_cup/ 
    --label_source "source_data" 
    --label_model1 "ACT" 
    --label_model2 "PI0" 
    --output ./save/output/reload_action_compare/ 
    --verbose
    '''
    print("Three-Way Action Data Comparison Tool")
    print("专门用于对比三个文件夹下HDF5文件的action数据")
    print("- 源数据使用 slave_left_arm 键名")
    print("- 模型1和模型2使用 left_arm 键名")
    print("- 支持自动匹配测试数据中包含的HDF5文件")
    print()
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入文件夹
    if not os.path.exists(args.source_folder):
        print(f"错误: 源数据文件夹不存在: {args.source_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.model1_folder):
        print(f"错误: 模型1文件夹不存在: {args.model1_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.model2_folder):
        print(f"错误: 模型2文件夹不存在: {args.model2_folder}")
        sys.exit(1)
    
    # 执行三方对比分析
    compare_three_folders(
        source_folder=args.source_folder,
        model1_folder=args.model1_folder,
        model2_folder=args.model2_folder,
        label_source=args.label_source,
        label_model1=args.label_model1,
        label_model2=args.label_model2,
        output_dir=args.output,
        verbose=args.verbose
    )
