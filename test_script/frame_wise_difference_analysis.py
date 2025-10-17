#!/usr/bin/env python3
"""
逐帧差异分析工具 - 逐文件计算模型和源数据的动作差异，统计相同帧位置的平均差和方差
专门用于分析两个模型与源数据之间的逐帧差异，生成详细的统计曲线图

主要功能：
1. 逐帧计算每个对应HDF5文件之间的动作差异（绝对差值）
2. 统计所有相同帧位置的差异值，计算平均和方差
3. 生成每帧平均差随时间变化的曲线图
4. 生成每帧方差随时间变化的曲线图
5. 保存CSV格式的曲线数据和维度汇总统计
6. 支持多种HDF5文件格式的自动匹配

计算方法：
- 对于每个文件对(source_i, model_i)，计算逐帧绝对差值
- 对于每个时间步t，收集所有文件在该时间步的差异值
- 计算该时间步所有差异值的平均值和方差

使用方法:
    conda activate telerobot
    python frame_wise_difference_analysis.py --source_folder /path/to/source --model1_folder /path/to/model1 --model2_folder /path/to/model2
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


def calculate_frame_wise_differences_pairwise(source_actions, model1_actions, model2_actions, verbose=False):
    """
    逐帧计算每个HDF5文件对之间的动作差异，然后统计相同帧位置的差异
    
    Parameters:
        source_actions: 源数据action列表
        model1_actions: 模型1 action列表  
        model2_actions: 模型2 action列表
        verbose: 是否输出详细信息
        
    Returns:
        dict: 包含差异分析结果的字典
    """
    if not source_actions:
        if verbose:
            print("源数据为空，无法计算差异")
        return {}
    
    # 确定要比较的文件数量（取最小值）
    n_files = len(source_actions)
    if model1_actions:
        n_files = min(n_files, len(model1_actions))
    if model2_actions:
        n_files = min(n_files, len(model2_actions))
    
    if n_files == 0:
        if verbose:
            print("没有可比较的文件")
        return {}
    
    # 找到所有文件的共同维度和时间步长
    all_actions = source_actions[:n_files]
    if model1_actions:
        all_actions.extend(model1_actions[:n_files])
    if model2_actions:
        all_actions.extend(model2_actions[:n_files])
    
    min_timesteps = min([data.shape[0] for data in all_actions])
    min_dims = min([data.shape[1] for data in all_actions])
    
    if verbose:
        print(f"将比较 {n_files} 个文件对")
        print(f"共同维度: {min_dims}, 共同时间步长: {min_timesteps}")
    
    results = {
        'timesteps': min_timesteps,
        'dimensions': min_dims,
        'n_files': n_files,
        'differences': {},
        'statistics': {}
    }
    
    # 计算模型1与源数据的逐文件差异
    if model1_actions and len(model1_actions) >= n_files:
        if verbose:
            print("计算模型1与源数据的逐文件差异...")
        
        # 存储每个文件对的差异 - shape: (n_files, timesteps, dims)
        file_differences_model1 = []
        
        for i in range(n_files):
            source_data = source_actions[i][:min_timesteps, :min_dims]
            model1_data = model1_actions[i][:min_timesteps, :min_dims]
            
            # 计算绝对差值
            diff = np.abs(model1_data - source_data)
            file_differences_model1.append(diff)
        
        # 转换为numpy数组 - shape: (n_files, timesteps, dims)
        file_differences_model1 = np.stack(file_differences_model1, axis=0)
        
        # 计算每帧的统计量（在所有文件上）
        mean_diff_per_frame = np.mean(file_differences_model1, axis=(0, 2))  # 对文件和维度求平均
        var_diff_per_frame = np.var(file_differences_model1, axis=(0, 2))    # 对文件和维度求方差
        
        # 计算每个维度的统计量（在所有文件和时间步上）
        mean_diff_per_dim = np.mean(file_differences_model1, axis=(0, 1))    # 对文件和时间步求平均
        var_diff_per_dim = np.var(file_differences_model1, axis=(0, 1))      # 对文件和时间步求方差
        
        results['differences']['model1'] = file_differences_model1
        results['statistics']['model1'] = {
            'mean_diff_per_dim': mean_diff_per_dim,
            'var_diff_per_dim': var_diff_per_dim,
            'mean_diff_per_frame': mean_diff_per_frame,
            'var_diff_per_frame': var_diff_per_frame,
            'overall_mean_diff': np.mean(file_differences_model1),
            'overall_var_diff': np.var(file_differences_model1)
        }
    
    # 计算模型2与源数据的逐文件差异
    if model2_actions and len(model2_actions) >= n_files:
        if verbose:
            print("计算模型2与源数据的逐文件差异...")
        
        # 存储每个文件对的差异 - shape: (n_files, timesteps, dims)
        file_differences_model2 = []
        
        for i in range(n_files):
            source_data = source_actions[i][:min_timesteps, :min_dims]
            model2_data = model2_actions[i][:min_timesteps, :min_dims]
            
            # 计算绝对差值
            diff = np.abs(model2_data - source_data)
            file_differences_model2.append(diff)
        
        # 转换为numpy数组 - shape: (n_files, timesteps, dims)
        file_differences_model2 = np.stack(file_differences_model2, axis=0)
        
        # 计算每帧的统计量（在所有文件上）
        mean_diff_per_frame = np.mean(file_differences_model2, axis=(0, 2))  # 对文件和维度求平均
        var_diff_per_frame = np.var(file_differences_model2, axis=(0, 2))    # 对文件和维度求方差
        
        # 计算每个维度的统计量（在所有文件和时间步上）
        mean_diff_per_dim = np.mean(file_differences_model2, axis=(0, 1))    # 对文件和时间步求平均
        var_diff_per_dim = np.var(file_differences_model2, axis=(0, 1))      # 对文件和时间步求方差
        
        results['differences']['model2'] = file_differences_model2
        results['statistics']['model2'] = {
            'mean_diff_per_dim': mean_diff_per_dim,
            'var_diff_per_dim': var_diff_per_dim,
            'mean_diff_per_frame': mean_diff_per_frame,
            'var_diff_per_frame': var_diff_per_frame,
            'overall_mean_diff': np.mean(file_differences_model2),
            'overall_var_diff': np.var(file_differences_model2)
        }
    
    if verbose:
        print("逐文件差异计算完成:")
        for model_name, stats in results['statistics'].items():
            print(f"  {model_name.upper()}:")
            print(f"    总体平均差异: {stats['overall_mean_diff']:.6f}")
            print(f"    总体差异方差: {stats['overall_var_diff']:.6f}")
    
    return results


def create_difference_curves(diff_results, labels, output_dir, verbose=False):
    """
    创建差异分析的曲线图 - 显示每帧的平均差和方差随时间变化
    现在分别保存为两个独立的图表文件
    
    Parameters:
        diff_results: 差异分析结果
        labels: 标签字典，包含source, model1, model2的标签
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    if not diff_results or 'statistics' not in diff_results:
        if verbose:
            print("没有有效的差异分析结果")
        return
    
    # 设置matplotlib和seaborn样式
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 0  # 设置为0以移除标题
    plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小放大2号
    plt.rcParams['xtick.labelsize'] = 14  # 横坐标刻度标签字体大小放大2号
    plt.rcParams['ytick.labelsize'] = 14  # 纵坐标刻度标签字体大小放大2号
    sns.set_style("whitegrid")
    
    dimensions = diff_results['dimensions']
    timesteps = diff_results['timesteps']
    time_axis = range(timesteps)
    
    # 定义颜色方案 - 使用莫兰迪色系中对比明显的颜色
    colors = {
        'model1': '#D4A574',    # 莫兰迪暖棕色 - 模型1
        'model2': '#7B9AAF'     # 莫兰迪蓝灰色 - 模型2
    }
    
    # 1. 创建并保存每帧平均差曲线
    plt.figure(figsize=(16, 8))
    ax1 = plt.gca()
    
    for model_key in ['model1', 'model2']:
        if model_key in diff_results['statistics']:
            model_label = labels.get(model_key, model_key.upper())
            stats = diff_results['statistics'][model_key]
            
            # 每帧的平均差异（所有维度的平均）
            mean_diff_per_frame = stats['mean_diff_per_frame']
            
            sns.lineplot(x=time_axis, y=mean_diff_per_frame, 
                        ax=ax1, label=model_label, 
                        linewidth=3.5, color=colors[model_key])
    
    ax1.set_xlabel('Time Step (Frame)')
    ax1.set_ylabel('平均差（rad）')  # 设置Y轴标题为平均差（rad）
    ax1.set_title('')   # 移除上方标题
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 保存平均差曲线图
    mean_curves_path = os.path.join(output_dir, 'frame_wise_mean_difference.png')
    plt.savefig(mean_curves_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存每帧平均差曲线图: {mean_curves_path}")
    
    # 2. 创建并保存每帧方差曲线
    plt.figure(figsize=(16, 8))
    ax2 = plt.gca()
    
    for model_key in ['model1', 'model2']:
        if model_key in diff_results['statistics']:
            model_label = labels.get(model_key, model_key.upper())
            stats = diff_results['statistics'][model_key]
            
            # 每帧的差异方差（所有维度的方差）
            var_diff_per_frame = stats['var_diff_per_frame']
            
            sns.lineplot(x=time_axis, y=var_diff_per_frame, 
                        ax=ax2, label=model_label, 
                        linewidth=3.5, color=colors[model_key])
    
    ax2.set_xlabel('Time Step (Frame)')
    ax2.set_ylabel('方差（rad）')  # 设置Y轴标题为方差（rad）
    ax2.set_title('')   # 移除上方标题
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 保存方差曲线图
    var_curves_path = os.path.join(output_dir, 'frame_wise_variance_difference.png')
    plt.savefig(var_curves_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存每帧方差曲线图: {var_curves_path}")
    
    # 同时保存CSV格式的数据
    save_csv_curves(diff_results, labels, output_dir, verbose)


def save_csv_curves(diff_results, labels, output_dir, verbose=False):
    """
    保存CSV格式的差异分析曲线数据
    
    Parameters:
        diff_results: 差异分析结果
        labels: 标签字典
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    timesteps = diff_results['timesteps']
    
    # 准备数据
    models_data = {}
    for model_key in ['model1', 'model2']:
        if model_key in diff_results['statistics']:
            model_label = labels.get(model_key, model_key.upper())
            stats = diff_results['statistics'][model_key]
            models_data[model_label] = stats
    
    if not models_data:
        return
    
    # 创建每帧平均差DataFrame
    mean_diff_df_data = {'Frame': list(range(timesteps))}
    for model_name, stats in models_data.items():
        mean_diff_df_data[f'{model_name}_Mean_Diff'] = stats['mean_diff_per_frame']
    
    mean_diff_df = pd.DataFrame(mean_diff_df_data)
    
    # 创建每帧方差DataFrame
    var_diff_df_data = {'Frame': list(range(timesteps))}
    for model_name, stats in models_data.items():
        var_diff_df_data[f'{model_name}_Var_Diff'] = stats['var_diff_per_frame']
    
    var_diff_df = pd.DataFrame(var_diff_df_data)
    
    # 保存CSV文件
    mean_csv_path = os.path.join(output_dir, 'frame_wise_mean_diff_curves.csv')
    var_csv_path = os.path.join(output_dir, 'frame_wise_var_diff_curves.csv')
    
    mean_diff_df.to_csv(mean_csv_path, index=False)
    var_diff_df.to_csv(var_csv_path, index=False)
    
    if verbose:
        print(f"保存每帧平均差曲线CSV: {mean_csv_path}")
        print(f"保存每帧方差曲线CSV: {var_csv_path}")

def save_csv_summary(diff_results, labels, output_dir, verbose=False):
    """
    保存CSV格式的差异分析汇总表格（按维度统计）
    
    Parameters:
        diff_results: 差异分析结果
        labels: 标签字典
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    dimensions = diff_results['dimensions']
    
    # 准备数据
    models_data = {}
    for model_key in ['model1', 'model2']:
        if model_key in diff_results['statistics']:
            model_label = labels.get(model_key, model_key.upper())
            stats = diff_results['statistics'][model_key]
            models_data[model_label] = stats
    
    if not models_data:
        return
    
    # 创建维度汇总DataFrame
    summary_data = {'Dimension': [f'Dim {i+1}' for i in range(dimensions)]}
    
    for model_name, stats in models_data.items():
        summary_data[f'{model_name}_Mean_Diff'] = stats['mean_diff_per_dim']
        summary_data[f'{model_name}_Var_Diff'] = stats['var_diff_per_dim']
    
    # 添加Overall行
    overall_row = {'Dimension': 'Overall'}
    for model_name, stats in models_data.items():
        overall_row[f'{model_name}_Mean_Diff'] = stats['overall_mean_diff']
        overall_row[f'{model_name}_Var_Diff'] = stats['overall_var_diff']
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
    
    # 保存CSV文件
    summary_csv_path = os.path.join(output_dir, 'frame_wise_summary_by_dimension.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    if verbose:
        print(f"保存维度汇总CSV: {summary_csv_path}")


def analyze_frame_wise_differences(source_folder, model1_folder, model2_folder, 
                                 label_source="Source Data", label_model1="Model 1", label_model2="Model 2",
                                 output_dir="output", verbose=False):
    """
    分析三个文件夹中模型与源数据的逐文件逐帧差异
    
    新的计算方法：
    1. 逐文件计算每个对应HDF5文件之间的动作差异
    2. 对于每个时间步，统计所有文件在该时间步的差异值
    3. 计算每个时间步差异值的平均和方差
    
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
        print("开始逐帧差异分析...")
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
        print(f"\n将分析 {len(common_file_triplets)} 组匹配的文件")
        print("="*80)
    
    # 分别收集三个文件夹的数据
    source_files_filter = [triplet[0] for triplet in common_file_triplets]  # 源数据文件
    model1_files_filter = [triplet[1] for triplet in common_file_triplets]  # 模型1文件
    model2_files_filter = [triplet[2] for triplet in common_file_triplets]  # 模型2文件
    
    data_source = collect_action_data_from_folder(source_folder, 'slave_left_arm', label_source, verbose, file_filter=source_files_filter)
    data_model1 = collect_action_data_from_folder(model1_folder, 'left_arm', label_model1, verbose, file_filter=model1_files_filter)
    data_model2 = collect_action_data_from_folder(model2_folder, 'left_arm', label_model2, verbose, file_filter=model2_files_filter)
    
    if not data_source.get('actions'):
        print("没有找到有效的源数据")
        return
    
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
        print("没有有效的action数据或没有共同的维度")
        return
    
    if verbose:
        print(f"将分析共同的 {common_dims} 个维度")
    
    # 使用新的逐文件差异计算方法
    diff_results = calculate_frame_wise_differences_pairwise(
        data_source.get('actions', []), 
        data_model1.get('actions', []), 
        data_model2.get('actions', []), 
        verbose
    )
    
    if not diff_results:
        print("无法计算差异分析")
        return
    
    # 创建标签字典
    labels = {
        'source': label_source,
        'model1': label_model1,
        'model2': label_model2
    }
    
    # 生成差异分析曲线图（现在会分别保存两张图）
    create_difference_curves(diff_results, labels, output_dir, verbose)
    
    # 同时保存维度汇总表格
    save_csv_summary(diff_results, labels, output_dir, verbose)
    
    if verbose:
        print("\n" + "="*80)
        print("逐文件逐帧差异分析完成！")
        print(f"结果保存在: {output_dir}")
        print(f"源数据 ({label_source}): {len(data_source.get('actions', []))} 个文件")
        print(f"模型1 ({label_model1}): {len(data_model1.get('actions', []))} 个文件")
        print(f"模型2 ({label_model2}): {len(data_model2.get('actions', []))} 个文件")
        print(f"分析了 {diff_results.get('n_files', 0)} 个文件对的逐帧差异")
        print(f"已分别保存平均差曲线和方差曲线为独立图表文件")
        print("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="逐帧差异分析工具 - 逐文件计算模型和源数据的动作差异，统计相同帧位置的平均差和方差",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    conda activate telerobot
    python frame_wise_difference_analysis.py --source_folder /path/to/source --model1_folder /path/to/model1 --model2_folder /path/to/model2
    python frame_wise_difference_analysis.py --source_folder save/pick_place_cup/ --model1_folder test/act_pick_place_cup/ --model2_folder test/pi0_pick_place_cup/ --verbose
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
                       help="模型1的标签 (默认: ACT)")
    parser.add_argument("--label_model2", type=str, default="Model 2",
                       help="模型2的标签 (默认: Pi0)")
    parser.add_argument("--output", type=str, default="save/output/frame_wise_difference_analysis",
                       help="输出目录 (默认: save/output/frame_wise_difference_analysis)")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细信息")
    
    return parser.parse_args()


if __name__ == "__main__":
    '''
    python test_script/frame_wise_difference_analysis.py \
    --source_folder ./save/pick_place_cup/ \
    --model1_folder ./test/reload_model_actions/act_pick_place_cup/ \
    --model2_folder ./test/reload_model_actions/pi0_pick_place_cup/ \
    --label_source "Human Demo" \
    --label_model1 "ACT Model" \
    --label_model2 "PI0 Model" \
    --output ./save/output/my_frame_analysis/ \
    --verbose
    Alternatively, run it using a script.
    bash test_script/run_frame_analysis.sh
    '''
    print("Frame-wise Difference Analysis Tool")
    print("专门用于分析模型与源数据的逐文件逐帧差异")
    print("- 逐文件计算每个对应HDF5文件之间的动作差异")
    print("- 统计所有相同帧位置的差异值，计算平均和方差")
    print("- 生成平均差和方差随时间变化的曲线图")
    print("- 支持CSV格式导出曲线数据和维度汇总统计")
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
    
    # 执行逐帧差异分析
    analyze_frame_wise_differences(
        source_folder=args.source_folder,
        model1_folder=args.model1_folder,
        model2_folder=args.model2_folder,
        label_source=args.label_source,
        label_model1=args.label_model1,
        label_model2=args.label_model2,
        output_dir=args.output,
        verbose=args.verbose
    )
