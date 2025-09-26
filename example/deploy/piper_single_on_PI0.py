import sys
sys.path.append('./')

# from my_robot.test_robot import TestRobot
from my_robot.agilex_piper_single_base import PiperSingle
from utils.bisocket import BiSocket
from utils.data_handler import debug_print, is_enter_pressed, hdf5_groups_to_dict
from my_robot.base_robot import dict_to_list
from policy.openpi.inference_model import PI0_SINGLE
from data.collect_any import CollectAny
import time
import numpy as np
import math
import pdb
import h5py
import os
condition = {
    "save_path": "./test/", 
    "task_name": "pi0_pick_place_cup", 
    "save_format": "hdf5", 
    "save_freq": 30,
    "collect_type": "teleop",
}
class ActionCollector:
    """专门用于收集和保存模型输出action的类，参考collect_any实现"""
    def __init__(self, save_path="save/model_actions", task_name="act_inference", resume=None):
        """
        Args:
            save_path: 保存路径
            task_name: 任务名称
            resume: resume标志位
                - None: 自动检测现有文件并提示用户选择（默认）
                - True: 叠加数据模式
                - False: 覆盖数据模式
        """
        self.episode = []
        self.episode_index = 0
        self.save_path = save_path
        self.task_name = task_name
        
        # 确保保存目录存在
        full_save_path = os.path.join(save_path, task_name)
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path)
        
        # 根据resume参数确定实际模式
        if resume is None:
            self.resume = self._auto_detect_and_choose_mode()
        else:
            self.resume = resume
        
        # 如果是叠加模式，检查并设置起始episode_index
        if self.resume:
            self._find_next_episode_index()
    
    def _auto_detect_and_choose_mode(self):
        """自动检测现有文件并让用户选择resume模式"""
        full_save_path = os.path.join(self.save_path, self.task_name)
        existing_files = []
        
        if os.path.exists(full_save_path):
            existing_files = [f for f in os.listdir(full_save_path) if f.endswith('.hdf5')]
        
        if existing_files:
            print(f"发现现有文件 {len(existing_files)} 个:")
            for f in existing_files[:5]:  # 只显示前5个文件
                print(f"  - {f}")
            if len(existing_files) > 5:
                print(f"  ... 还有 {len(existing_files) - 5} 个文件")
            
            print("\n请选择resume模式:")
            print("1. overwrite - 覆盖现有文件")
            print("2. append - 在现有文件基础上叠加数据")
            
            while True:
                choice = input("请输入选择 (1/2) [默认: 1]: ").strip()
                if choice == "2":
                    print("选择了叠加模式")
                    return True
                elif choice == "1" or choice == "":
                    print("选择了覆盖模式")
                    return False
                else:
                    print("无效选择，请输入 1 或 2")
        else:
            print("没有发现现有文件，使用默认覆盖模式")
            return False
    
    def _find_next_episode_index(self):
        """查找下一个可用的episode索引"""
        full_save_path = os.path.join(self.save_path, self.task_name)
        existing_episodes = []
        
        if os.path.exists(full_save_path):
            for filename in os.listdir(full_save_path):
                if filename.startswith("episode_") and filename.endswith(".hdf5"):
                    try:
                        # 提取episode号码
                        episode_num = int(filename.replace("episode_", "").replace(".hdf5", ""))
                        existing_episodes.append(episode_num)
                    except ValueError:
                        continue
        
        if existing_episodes:
            self.episode_index = max(existing_episodes) + 1
            print(f"叠加模式: 找到已存在的episodes，下一个episode将从 {self.episode_index} 开始")
        else:
            self.episode_index = 0
            print("叠加模式: 没有找到已存在的episodes，从episode 0开始")
    
    def _load_existing_episode(self, hdf5_path):
        """从现有的hdf5文件加载数据"""
        if not os.path.exists(hdf5_path):
            return []
        
        try:
            with h5py.File(hdf5_path, "r") as f:
                actions = f["action"]["model_output"][:]
                states = f["observations"]["qpos"][:]
                steps = f["metadata"]["steps"][:]
                timestamps = f["metadata"]["timestamps"][:]
                
                episode_data = []
                for i in range(len(actions)):
                    episode_data.append({
                        "step": steps[i],
                        "action": actions[i],
                        "state": states[i],
                        "timestamp": timestamps[i]
                    })
                
                print(f"从现有文件加载了 {len(episode_data)} 步数据")
                return episode_data
        except Exception as e:
            print(f"加载现有文件时出错: {e}")
            return []
    
    def collect(self, action, state, step):
        """收集一个时间步的数据（只保存动作和状态，不保存图像）"""
        episode_data = {
            "step": step,
            "action": np.array(action) if not isinstance(action, np.ndarray) else action,
            "state": np.array(state) if not isinstance(state, np.ndarray) else state,
            "timestamp": time.time()
        }
        self.episode.append(episode_data)
    
    def save_episode(self, custom_episode_name=None, force_overwrite=False):
        """保存当前episode到hdf5文件
        
        Args:
            custom_episode_name: 可选的自定义episode名称，如果提供则使用该名称而不是episode_index
            force_overwrite: 强制覆盖现有文件，忽略resume_mode设置
        """
        if len(self.episode) == 0:
            print("No data to save!")
            return
            
        full_save_path = os.path.join(self.save_path, self.task_name)
        
        # 如果提供了自定义名称，使用自定义名称；否则使用episode_index
        if custom_episode_name is not None:
            hdf5_path = os.path.join(full_save_path, f"{custom_episode_name}.hdf5")
        else:
            hdf5_path = os.path.join(full_save_path, f"episode_{self.episode_index}.hdf5")
        
        # 确定保存模式
        should_append = (self.resume and 
                        not force_overwrite and 
                        os.path.exists(hdf5_path))
        
        if should_append:
            # 叠加模式：先加载现有数据
            existing_data = self._load_existing_episode(hdf5_path)
            all_episode_data = existing_data + self.episode
            print(f"叠加模式: 合并了 {len(existing_data)} 步现有数据和 {len(self.episode)} 步新数据")
        else:
            # 覆盖模式：只使用当前数据
            all_episode_data = self.episode
            if os.path.exists(hdf5_path):
                print(f"覆盖模式: 将覆盖现有文件 {hdf5_path}")
        
        # 保存数据
        with h5py.File(hdf5_path, "w") as f:
            # 创建action组
            action_group = f.create_group("action")
            actions = np.array([ep["action"] for ep in all_episode_data])
            action_group.create_dataset("model_output", data=actions)
            
            # 创建observation组
            obs_group = f.create_group("observations")
            states = np.array([ep["state"] for ep in all_episode_data])
            obs_group.create_dataset("qpos", data=states)
            
            # 保存元数据
            metadata_group = f.create_group("metadata")
            steps = np.array([ep["step"] for ep in all_episode_data])
            timestamps = np.array([ep["timestamp"] for ep in all_episode_data])
            metadata_group.create_dataset("steps", data=steps)
            metadata_group.create_dataset("timestamps", data=timestamps)
        
        episode_name = custom_episode_name if custom_episode_name else f"episode_{self.episode_index}"
        mode_info = "叠加模式" if should_append else "覆盖模式"
        print(f"Episode {episode_name} saved to {hdf5_path} ({mode_info})")
        print(f"Total steps: {len(all_episode_data)}")
        
        # 重置episode并增加索引
        self.episode = []
        if custom_episode_name is None:  # 只有在使用默认命名时才增加索引
            self.episode_index += 1
    
    def reset(self):
        """重置收集器"""
        self.episode = []
    
def input_transform(data):
    has_left_arm = "left_arm" in data[0]
    has_right_arm = "right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": [0.0] * left_gripper_dim
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["sleft_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": [0.0] * right_gripper_dim
        }
        has_left_arm = True
    
    elif not has_left_arm and not has_right_arm:
        default_joint_dim = 6
        
        data[0]["left_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        data[0]["right_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        has_left_arm = True
        has_right_arm = True
    
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        # np.array(data[0]["slave_right_arm"]["joint"]).reshape(-1),
        # np.array(data[0]["slave_right_arm"]["gripper"]).reshape(-1)
    ])
    # print(state)
    img_arr = data[1]["cam_head"]["color"], data[1]["cam_wrist"]["color"]
    return img_arr, state

def output_transform(data):
    # print(data)
    joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),    # joint2
        (math.radians(-170), math.radians(0)),   # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),   # joint5
        (math.radians(-120), math.radians(120))    # joint6
        ]
    def clamp(value, min_val, max_val):
        """将值限制在[min_val, max_val]范围内"""
        return max(min_val, min(value, max_val))
    left_joints = [
        clamp(data[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    if data[6] < 0.12:
        data[6] = 0.0
    left_gripper = data[6]
    
    move_data = {
        "left_arm":{
            "joint": left_joints,
            "gripper": left_gripper,
        }
    }
    return move_data


if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    robot = PiperSingle()
    robot.set_up()
    model = PI0_SINGLE("/home/usst/kwj/GitCode/control_your_robot_jie/policy/openpi/checkpoints/25000/", "pick_place_cup")
    collection=CollectAny(condition=condition,start_episode=0,move_check=True,resume=True)
    robot.reset()
    time.sleep(1)
    # 创建action收集器
    # resume参数选项:
    # - None: 自动检测现有文件并提示用户选择（默认）
    # - True: 叠加数据模式
    # - False: 覆盖数据模式
    action_collector = ActionCollector(
        save_path="test/pick_place_cup_real/model_actions/", 
        task_name="pi0_pick_place_cup",
        resume=True  # 使用自动检测模式
    )
    max_step = 1000
    num_episode = 6
    for i in range(num_episode):
        step = 0
        
        # 重置所有信息
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()
        action_collector.reset()  # 重置action收集器
        
        # 初始化耗时统计
        inference_times = []
        
        # 等待允许执行推理指令, 按enter开始
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("start to inference...")
            else:
                print("waiting for start command...")
                time.sleep(1)

        # 开始逐条推理运行
        while step < max_step:
            data = robot.get()
            img_arr, state = input_transform(data)
            
            model.update_observation_window(img_arr, state)
            
            # 测量 model.get_action() 的耗时
            start_time = time.time()
            action_chunk = model.get_action()
            end_time = time.time()
            inference_time = end_time - start_time
            
            action_chunk = action_chunk[:30]
            
            # 记录和打印耗时信息
            inference_times.append(inference_time)
            # pdb.set_trace()
            for action in action_chunk:
                move_data = output_transform(action)
                robot.move({"arm": 
                            move_data
                        })
                # action_collector.collect(action, state, step)
                step += 1
                data=robot.get()
                collection.collect(data[0],None)
                time.sleep(1/30)
                print(f"Episode {i}, Step {step}/{max_step} completed.")
        
        # Episode完成，保存收集的action数据，使用源文件名作为episode名称
        # action_collector.save_episode(custom_episode_name=f"episode_{i}")
        
        # 打印耗时统计
        if inference_times:
            import numpy as np
            avg_time = np.mean(inference_times)
            
            print(f"\n=== Episode {i} 推理耗时统计 ===")
            print(f"总推理次数: {len(inference_times)}")
            print(f"平均耗时: {avg_time:.4f}秒 ({avg_time*1000:.2f}ms)")
        time.sleep(1)
        robot.reset()
        collection.write()
        print("finish episode", i)
    
    robot.reset()
