import sys
sys.path.append('./')

import os
import importlib
import argparse
import numpy as np
import time
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from utils.data_handler import debug_print, is_enter_pressed, hdf5_groups_to_dict
from my_robot.base_robot import dict_to_list

### =========THE PLACE YOU COULD MODIFY=========
# eval setting
DRAW = True
DRAW_DIR = "save/picture/"
MODEL_CHUNK_SIZE = 50
SKIP_FRAMWE = 20

def input_transform(data):
    has_left_arm = "slave_left_arm" in data[0]
    has_right_arm = "slave_right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["slave_left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["slave_right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": [0.0] * left_gripper_dim
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["slave_right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["slave_left_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": [0.0] * right_gripper_dim
        }
        has_left_arm = True
    
    elif not has_left_arm and not has_right_arm:
        default_joint_dim = 6
        
        data[0]["slave_left_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        data[0]["slave_right_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        has_left_arm = True
        has_right_arm = True
    
    state = np.concatenate([
        np.array(data[0]["slave_left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["slave_left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["slave_right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["slave_right_arm"]["gripper"]).reshape(-1)
    ])
    # print(state)
    img_arr = data[1]["slave_cam_head"]["color"], data[1]["slave_cam_wrist"]["color"]
    return img_arr, state

def compare_transform(data_chunk):
    actions = []

    for data in data_chunk[0]:
        # 检查是否只有左臂数据
        if "slave_left_arm" in data and "slave_right_arm" not in data:
            # 获取左臂数据
            left_joint = np.array(data["slave_left_arm"]["joint"]).reshape(-1)
            left_gripper = np.array(data["slave_left_arm"]["gripper"]).reshape(-1)
            
            # 复制左臂数据作为右臂数据，保持维度一致性
            right_joint = np.zeros_like(left_joint)  # 用零填充右臂关节数据
            right_gripper = np.zeros_like(left_gripper)  # 用零填充右臂夹爪数据
            
            # 如果模型期望14维输出 (7维左臂 + 7维右臂)，则复制左臂数据
            action = np.concatenate([
                left_joint,
                left_gripper,
                right_joint,
                right_gripper
            ])
        else:
            # 正常情况，包含左右臂数据
            action = np.concatenate([
                np.array(data["slave_left_arm"]["joint"]).reshape(-1),
                np.array(data["slave_left_arm"]["gripper"]).reshape(-1),
                np.array(data["slave_right_arm"]["joint"]).reshape(-1),
                np.array(data["slave_right_arm"]["gripper"]).reshape(-1)
            ])
        actions.append(action)

    return np.stack(actions)

def compute_similarity(action_chunk_pred, action_chunk_real):
    dist = np.linalg.norm(action_chunk_pred - action_chunk_real)
    sim = 1 / (1 + dist)
    return sim

def compute_statistics(all_pred_actions, all_real_actions, all_time_chunks):
    """
    计算所有episode的统计信息：平均轨迹、平均差异、方差等
    """
    # 找到最长的时间步数，用于对齐所有轨迹
    max_steps = 0
    for pred_actions in all_pred_actions:
        total_steps = sum(chunk.shape[0] for chunk in pred_actions)
        max_steps = max(max_steps, total_steps)
    
    # 获取动作维度
    action_dim = all_pred_actions[0][0].shape[1]
    
    # 初始化存储数组
    all_pred_aligned = []  # 存储对齐后的预测轨迹
    all_real_aligned = []  # 存储对齐后的真实轨迹
    
    # 对每个episode的轨迹进行拼接和对齐
    for i, (pred_chunks, real_chunks) in enumerate(zip(all_pred_actions, all_real_actions)):
        # 拼接当前episode的所有chunks
        pred_traj = np.concatenate(pred_chunks, axis=0)  # (total_steps, action_dim)
        real_traj = np.concatenate(real_chunks, axis=0)  # (total_steps, action_dim)
        
        # 对齐到相同长度（截断或填充）
        current_steps = pred_traj.shape[0]
        if current_steps < max_steps:
            # 用最后一个值填充
            pred_pad = np.tile(pred_traj[-1:], (max_steps - current_steps, 1))
            real_pad = np.tile(real_traj[-1:], (max_steps - current_steps, 1))
            pred_traj = np.concatenate([pred_traj, pred_pad], axis=0)
            real_traj = np.concatenate([real_traj, real_pad], axis=0)
        elif current_steps > max_steps:
            # 截断
            pred_traj = pred_traj[:max_steps]
            real_traj = real_traj[:max_steps]
        
        all_pred_aligned.append(pred_traj)
        all_real_aligned.append(real_traj)
    
    # 转换为numpy数组 (num_episodes, max_steps, action_dim)
    all_pred_aligned = np.stack(all_pred_aligned)
    all_real_aligned = np.stack(all_real_aligned)
    
    # 计算统计信息
    # 平均轨迹 (max_steps, action_dim)
    mean_pred_traj = np.mean(all_pred_aligned, axis=0)
    mean_real_traj = np.mean(all_real_aligned, axis=0)
    
    # 计算差异 (num_episodes, max_steps, action_dim)
    differences = all_pred_aligned - all_real_aligned
    
    # 每个step的平均差异和方差 (max_steps, action_dim)
    mean_diff = np.mean(differences, axis=0)
    var_diff = np.var(differences, axis=0)
    std_diff = np.std(differences, axis=0)
    
    # 每个step每个维度的绝对差异平均值 (max_steps, action_dim)
    mean_abs_diff = np.mean(np.abs(differences), axis=0)
    
    return {
        'mean_pred_traj': mean_pred_traj,
        'mean_real_traj': mean_real_traj,
        'mean_diff': mean_diff,
        'var_diff': var_diff,
        'std_diff': std_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_steps': max_steps,
        'action_dim': action_dim
    }

### =========THE PLACE YOU COULD MODIFY=========

class Replay:
    def __init__(self, hdf5_path) -> None:
        self.ptr = 0
        self.episode = dict_to_list(hdf5_groups_to_dict(hdf5_path))
    
    def get_data(self):
        try:
            data = self.episode[self.ptr], self.episode[self.ptr]
            data_chunk_end_ptr = min(len(self.episode), self.ptr+MODEL_CHUNK_SIZE)
            data_chunk = self.episode[self.ptr:data_chunk_end_ptr], self.episode[self.ptr:data_chunk_end_ptr]
            self.ptr += SKIP_FRAMWE
        except:
            return None, None
        return data, data_chunk

def get_class(import_name, class_name):
    try:
        class_module = importlib.import_module(import_name)
        debug_print("function", f"Module loaded: {class_module}", "DEBUG")
    except ModuleNotFoundError as e:
        raise SystemExit(f"ModuleNotFoundError: {e}")

    try:
        return_class = getattr(class_module, class_name)
        debug_print("function", f"Class found: {return_class}", "DEBUG")

    except AttributeError as e:
        raise SystemExit(f"AttributeError: {e}")
    except Exception as e:
        raise SystemExit(f"Unexpected error instantiating model: {e}")
    return return_class

def eval_once(model, episode):
    replay = Replay(episode)

    similaritys = []
    action_chunk_preds = []
    action_chunk_reals = []
    time_step_chunks = []
    while True:
        time_step_chunk = (replay.ptr, replay.ptr + MODEL_CHUNK_SIZE)
        data, data_chunk = replay.get_data()
        if data is None:
            break

        img_arr, state = input_transform(data)
        model.update_observation_window(img_arr, state)
        action_chunk_pred = model.get_action()
        action_chunk_real = compare_transform(data_chunk)
        if action_chunk_real.shape[0] != action_chunk_pred.shape[0]:
            action_chunk_pred = action_chunk_pred[:action_chunk_real.shape[0]]
        similarity = compute_similarity(action_chunk_pred, action_chunk_real)

        similaritys.append(similarity)
        action_chunk_preds.append(action_chunk_pred)
        action_chunk_reals.append(action_chunk_real)
        time_step_chunks.append(time_step_chunk)

    if DRAW:
        if not os.path.exists(DRAW_DIR):
            os.makedirs(DRAW_DIR)
        
        pic_name = os.path.basename(episode).split(".")[0] + ".png"

        plot_trajectories_subplots(action_chunk_preds, action_chunk_reals, time_step_chunks, os.path.join(DRAW_DIR, pic_name)) 
        # plot_trajectories_subplots([similarity], None, time_step_chunks, save_path = os.path.join(DRAW_DIR, "similarity_" + pic_name)) 

def plot_trajectories_subplots(trajA, trajB, time_intervals, save_path="traj_subplots.png"):
    """
    每个维度一个子图绘制轨迹A和B，多段轨迹，支持不同时间区间
    trajA, trajB: list of np.ndarray, 每段轨迹形状 (seg_len, num_dims)，可为None
    time_intervals: list of tuples (a, b)，对应每段轨迹的时间区间
    """
    sns.set_style("whitegrid")  # 使用 seaborn 风格

    # 检查至少有一条轨迹存在
    if trajA is None and trajB is None:
        print("No trajectories to plot.")
        return

    # 获取维度数量
    sample_traj = trajA if trajA is not None else trajB
    num_dims = sample_traj[0].shape[1]

    # 动态调整图像高度，确保有足够空间
    height_per_dim = max(3, 4 - num_dims * 0.1)  # 维度多时稍微减小高度
    fig, axes = plt.subplots(num_dims, 1, figsize=(16, height_per_dim*num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]

    # 定义颜色方案，与frame_wise_difference_analysis一致
    colors = {"A": "#1E90FF", "B": "#FFD700"}  # A为蓝色，B为黄色
    labels_dict = {"A": "A", "B": "B"}

    num_segments = len(time_intervals) if time_intervals is not None else len(sample_traj)

    # 为每个维度创建标签追踪
    legend_added = {"A": [False] * num_dims, "B": [False] * num_dims}
    
    for seg_idx in range(num_segments):
        a, b = time_intervals[seg_idx] if time_intervals is not None else (0, sample_traj[seg_idx].shape[0]-1)
        t = np.linspace(a, b, sample_traj[seg_idx].shape[0])

        if trajA is not None:
            segA = trajA[seg_idx]
            for dim in range(num_dims):
                # 只在该维度第一次绘制时添加标签
                label = labels_dict["A"] if not legend_added["A"][dim] else ""
                sns.lineplot(x=t, y=segA[:, dim], ax=axes[dim], 
                            color=colors["A"], alpha=0.8, linewidth=3.5, 
                            label=label)
                legend_added["A"][dim] = True
                
        if trajB is not None:
            segB = trajB[seg_idx]
            for dim in range(num_dims):
                # 只在该维度第一次绘制时添加标签
                label = labels_dict["B"] if not legend_added["B"][dim] else ""
                sns.lineplot(x=t, y=segB[:, dim], ax=axes[dim], 
                            color=colors["B"], alpha=0.8, linewidth=3.5, 
                            label=label)
                legend_added["B"][dim] = True

    # 设置每个子图的标题和标签
    for dim in range(num_dims):
        axes[dim].set_ylabel(f"Dim {dim}", fontsize=14)
        axes[dim].grid(True, alpha=0.3)
        # 移除顶部和右侧边框
        axes[dim].spines['top'].set_visible(False)
        axes[dim].spines['right'].set_visible(False)
        # 每个子图都添加图例
        axes[dim].legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)
        # 设置刻度标签字体大小
        axes[dim].tick_params(axis='both', which='major', labelsize=14)
    
    # 设置x轴标签
    axes[-1].set_xlabel("Time Step (Frame)", fontsize=16)

    # 使用subplots_adjust代替tight_layout，避免警告
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.95, hspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved trajectory figure to {save_path}")

    
def init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the task") 
    parser.add_argument("--model_class", type=str, required=True, help="Name of the model class")
    parser.add_argument("--model_path", type=str, required=True, help="model path, e.g., policy/RDT/checkpoints/checkpoint-10000")
    parser.add_argument("--task_name", type=str, required=True, help="task name, read intructions from task_instuctions/{task_name}.json")
    parser.add_argument("--data_path", type=str, required=True, help="the data you want to eval")
    parser.add_argument("--episode_num", type=int, required=False,default=10, help="how many episode you want to eval")
    
    args = parser.parse_args()
    model_name = args.model_name
    model_class = args.model_class
    model_path = args.model_path
    task_name = args.task_name
    data_path = args.data_path
    episode_num = args.episode_num

    model_class = get_class(f"policy.{model_name}.inference_model", model_class)
    model = model_class(model_path, task_name)

    if os.path.isfile(data_path):
        return model, [data_path]   
    else:
        all_files = glob.glob(os.path.join(data_path, "*.hdf5"))
        if episode_num > len(all_files):
            raise IndexError(f"episode_num > data_num : {episode_num} > len(all_files)")

        # 随机选取
        episodes = random.sample(all_files, episode_num)

        return model, episodes

if __name__ == "__main__":
    os.environ["INFO_LEVEL"] = "INFO" # DEBUG , INFO, ERROR

    model, episodes = init()

    for episode in episodes:
        eval_once(model, episode)
