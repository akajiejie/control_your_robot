#!/usr/bin/env python3
"""
对比三个文件夹下HDF5文件中的平均action曲线
- 文件夹1：源数据，使用 slave_left_arm 键名
- 文件夹2：模型1推理数据，使用 left_arm 键名  
- 文件夹3：模型2推理数据，使用 left_arm 键名
专门为 telerobot conda 环境设计

使用方法:
    conda activate telerobot
    python action_average_compare.py --source_folder /path/to/source --model1_folder /path/to/model1 --model2_folder /path/to/model2
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
                    'observations/left_arm/qpos'
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


def collect_action_data_from_folder(folder_path, key_name, label, verbose=False):
    """
    从文件夹中收集所有HDF5文件的action数据
    
    Parameters:
        folder_path: 文件夹路径
        key_name: HDF5文件中的键名
        label: 数据源标签
        verbose: 是否输出详细信息
        
    Returns:
        list: 收集到的所有action数据
    """
    if not os.path.exists(folder_path):
        if verbose:
            print(f"文件夹不存在: {folder_path}")
        return []
    
    # 查找所有HDF5文件
    hdf5_files = []
    for ext in ['*.hdf5', '*.h5']:
        hdf5_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not hdf5_files:
        if verbose:
            print(f"在文件夹 {folder_path} 中未找到HDF5文件")
        return []
    
    if verbose:
        print(f"在 {folder_path} 中找到 {len(hdf5_files)} 个HDF5文件")
        print(f"使用键名: {key_name}")
    
    collected_actions = []
    
    # 按文件名排序，确保处理顺序一致
    hdf5_files.sort()
    
    # 处理每个HDF5文件
    success_count = 0
    for hdf5_file in tqdm(hdf5_files, desc=f"Reading {label} files", disable=not verbose):
        action_data = read_action_data_from_hdf5(hdf5_file, key_name, verbose=False)
        
        if action_data is not None:
            collected_actions.append(action_data)
            success_count += 1
    
    if verbose:
        print(f"{label} - 成功读取: {success_count}/{len(hdf5_files)} 个文件")
        if collected_actions:
            shapes = [data.shape for data in collected_actions]
            print(f"数据形状: {shapes[:5]}...")  # 只显示前5个
    
    return collected_actions


def calculate_average_action_curve(action_list, verbose=False):
    """
    计算多个action数据的平均曲线
    
    Parameters:
        action_list: action数据列表
        verbose: 是否输出详细信息
        
    Returns:
        numpy.ndarray: 平均曲线数据，形状为 (timesteps, action_dims)
    """
    if not action_list:
        return None
    
    # 找到最小的时间步长和action维度，以便对齐数据
    min_timesteps = min([data.shape[0] for data in action_list])
    min_dims = min([data.shape[1] for data in action_list])
    
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


def plot_average_curves_comparison(avg_actions_source, avg_actions_model1, avg_actions_model2, 
                                 label_source, label_model1, label_model2,
                                 n_files_source, n_files_model1, n_files_model2, 
                                 output_dir, verbose=False):
    """
    绘制三个数据源的平均action曲线对比图
    使用seaborn绘图，标签统一放在右上角
    
    Parameters:
        avg_actions_source: 源数据的平均曲线
        avg_actions_model1: 模型1的平均曲线
        avg_actions_model2: 模型2的平均曲线
        label_source: 源数据的标签
        label_model1: 模型1的标签
        label_model2: 模型2的标签
        n_files_source: 源数据的文件数量
        n_files_model1: 模型1的文件数量
        n_files_model2: 模型2的文件数量
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    # 设置seaborn样式和matplotlib参数
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    
    # 检查是否有有效数据
    valid_data = [avg_actions_source, avg_actions_model1, avg_actions_model2]
    valid_data = [data for data in valid_data if data is not None]
    
    if not valid_data:
        if verbose:
            print("没有有效的平均曲线数据")
        return
    
    # 确定共同的维度数
    common_dims = min([data.shape[1] for data in valid_data])
    
    if common_dims == 0:
        if verbose:
            print("没有共同的action维度")
        return
    
    if verbose:
        print(f"将绘制 {common_dims} 个维度的平均曲线对比")
    
    # 创建图形 - 调整布局避免重叠
    cols = 3
    rows = (common_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    
    # 创建标题 - 更紧凑的格式
    title = f'Average Action Curves Comparison\n'
    title += f'Source: {n_files_source} files | Model1: {n_files_model1} files | Model2: {n_files_model2} files'
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
                       ax=ax, label=f'{label_source}', 
                       linewidth=2.5, color=colors['source'])
        
        # 绘制模型1的平均曲线
        if avg_actions_model1 is not None and dim_idx < avg_actions_model1.shape[1]:
            time_steps = range(len(avg_actions_model1))
            sns.lineplot(x=time_steps, y=avg_actions_model1[:, dim_idx], 
                       ax=ax, label=f'{label_model1}', 
                       linewidth=2.5, color=colors['model1'])
        
        # 绘制模型2的平均曲线
        if avg_actions_model2 is not None and dim_idx < avg_actions_model2.shape[1]:
            time_steps = range(len(avg_actions_model2))
            sns.lineplot(x=time_steps, y=avg_actions_model2[:, dim_idx], 
                       ax=ax, label=f'{label_model2}', 
                       linewidth=2.5, color=colors['model2'])
        
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
    output_path = os.path.join(output_dir, 'three_way_average_curves_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    if verbose:
        print(f"保存三方平均曲线对比图: {output_path}")


def compare_three_folders_average_curves(source_folder, model1_folder, model2_folder, 
                                        label_source="Source Data", label_model1="Model 1", label_model2="Model 2",
                                        output_dir="output", verbose=False):
    """
    对比三个文件夹中HDF5文件的平均action曲线
    
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
        print("开始对比三个文件夹的平均action曲线...")
        print(f"源数据文件夹: {source_folder} (键名: slave_left_arm)")
        print(f"模型1文件夹: {model1_folder} (键名: left_arm)")
        print(f"模型2文件夹: {model2_folder} (键名: left_arm)")
        print(f"输出目录: {output_dir}")
        print("="*80)
    
    # 收集三个文件夹的action数据
    actions_source = collect_action_data_from_folder(source_folder, 'slave_left_arm', label_source, verbose)
    actions_model1 = collect_action_data_from_folder(model1_folder, 'left_arm', label_model1, verbose)
    actions_model2 = collect_action_data_from_folder(model2_folder, 'left_arm', label_model2, verbose)
    
    if not actions_source and not actions_model1 and not actions_model2:
        print("三个文件夹都没有找到有效的action数据")
        return
    
    if verbose:
        print("\n" + "="*60)
        print("计算平均曲线...")
    
    # 计算平均曲线
    avg_actions_source = calculate_average_action_curve(actions_source, verbose) if actions_source else None
    avg_actions_model1 = calculate_average_action_curve(actions_model1, verbose) if actions_model1 else None
    avg_actions_model2 = calculate_average_action_curve(actions_model2, verbose) if actions_model2 else None
    
    if verbose:
        print("\n" + "="*60)
        print("生成对比图...")
    
    # 绘制对比图
    plot_average_curves_comparison(
        avg_actions_source, avg_actions_model1, avg_actions_model2,
        label_source, label_model1, label_model2,
        len(actions_source), len(actions_model1), len(actions_model2),
        output_dir, verbose
    )
    
    if verbose:
        print("\n" + "="*80)
        print("三方对比分析完成！")
        print(f"结果保存在: {output_dir}")
        print(f"源数据 ({label_source}): {len(actions_source)} 个文件")
        print(f"模型1 ({label_model1}): {len(actions_model1)} 个文件")
        print(f"模型2 ({label_model2}): {len(actions_model2)} 个文件")
        print("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="对比三个文件夹下HDF5文件中的平均action曲线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    conda activate telerobot
    python action_average_compare.py --source_folder /path/to/source --model1_folder /path/to/model1 --model2_folder /path/to/model2
    python action_average_compare.py --source_folder save/pick_place_cup/ --model1_folder test/model1/ --model2_folder test/model2/ --verbose
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
    parser.add_argument("--output", type=str, default="save/output/three_way_comparison",
                       help="输出目录 (默认: save/output/three_way_comparison)")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细信息")
    
    return parser.parse_args()





if __name__ == "__main__":
    '''
    使用示例
    python test_script/action_average_compare.py 
    --source_folder ./save/pick_place_cup/ 
    --model1_folder ./test/act_pick_place_cup/ 
    --model2_folder ./test/pi0_pick_place_cup/ 
    --label_source "source_data" 
    --label_model1 "ACT" 
    --label_model2 "PI0" 
    --output ./save/output/test_action_compare/ 
    --verbose
    '''
    print("Three-Way Action Average Curves Comparison Tool")
    print("专门用于对比三个文件夹下HDF5文件的平均action曲线")
    print("- 源数据使用 slave_left_arm 键名")
    print("- 模型1和模型2使用 left_arm 键名")
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
    compare_three_folders_average_curves(
        source_folder=args.source_folder,
        model1_folder=args.model1_folder,
        model2_folder=args.model2_folder,
        label_source=args.label_source,
        label_model1=args.label_model1,
        label_model2=args.label_model2,
        output_dir=args.output,
        verbose=args.verbose
    )
