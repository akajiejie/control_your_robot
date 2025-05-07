import numpy as np
import os
import h5py
import cv2
import json

'''
condition:
    image_keys: list[str], 视角名称 [front_image, left_wrist_image, right_wrist_image]
    arm_type: str, 机械臂类型
    state_is_joint: bool, 关节角度是否为state
    is_action: bool, 是否包含action
    is_dual: bool, 是否为双臂
    save_right_now: bool, 是否在当前时刻保存数据
    save_depth: bool, 是否保存深度图
    save_path: str, 保存路径
    task_name: str, 任务名称
    save_format: str, 保存格式
    save_interval: int, 保存频率
    
image_map:
    # if is_dual is True, 想要保存的完整三个视角必须全部正确设置,否则只设置你要的视角名称的映射就行
    "cam_head": "your image_key",
    "cam_left_wrist": "your image_key",
    "cam_right_wrist": "your image_key",
    # if is_dual is False, 想要保存的完整两个视角必须正确设置,否则只设置你要的视角名称的映射就行
    "cam_head": "your image_key",
    "cam_wrist": "your image_key",
    ......
    # 允许额外设置映射, 如果设置便会存储
    "extra_image": "your image_key",
'''
DUAL_ARM_CAMERA_NAME = ["front_image", "left_wrist_image", "right_wrist_image"]
SINGLE_ARM_CAMERA_NAME = ["front_image", "wrist_image"]

class collection:
    def __init__(self, condition, image_map=None, start_episode=0):
        if image_map is None:
            if condition["is_dual"]:
                image_map = {
                    "cam_head": "front_image",
                    "cam_left_wrist": "left_wrist_image",
                    "cam_right_wrist": "right_wrist_image",
                }
            else:
                image_map = {
                    "cam_head": "front_image",
                    "cam_wrist": "wrist_image",
                }
        
        self.image_map = image_map
        self.condition = condition
        self.episode = []
        self.episode_index = start_episode
    
    def collect(self, data):
        # 保存关节角度、位姿、夹爪状态
        if  not self.condition["is_dual"]:
            episode_data = {
                "joint": data["joint"],
                "pose": data["pose"],
                "gripper": data["gripper"],
                }
            if self.condition["is_action"]:
                episode_data["action"] = data["action"]
        else:
            episode_data = {
                "joint_left": data["joint_left"],
                "joint_right": data["joint_right"],
                "pose_left": data["pose_left"],
                "pose_right": data["pose_right"],
                "gripper_left": data["gripper_left"],
                "gripper_right": data["gripper_right"],
            }
            if self.condition["is_action"]:
                episode_data["action_left"] = data["action_left"]
                episode_data["action_right"] = data["action_right"]
        # 保存图像
        for key in self.condition["image_keys"]:
            try:
                episode_data[key+"_color"] = np.array(data[key+"_color"])
                if self.condition["save_depth"]:
                    episode_data[key+"_depth"] = np.array(data[key+"_depth"])
            except:
                raise ValueError(f"input data dont have key: {key}")

        self.episode.append(episode_data)
        if self.condition["save_right_now"] == true:
            self.write(only_end=True)
    
    # 根据映射图,将所有图像进行编码
    def encode_images(self):
        enc_images = {}
        for key in self.image_map.keys():
            key_images = get_images(self.episode, self.image_map[key])
            if not key_images:
                raise ValueError(f"image_key: {key} is empty")
            else:
                key_images_enc, key_images_len = images_encoding(key_images)
                enc_images[key] = {"enc_data": key_images_enc, "len": key_images_len}
        return enc_images
    
    def get_state(self):
        if self.condition["is_dual"]:
            gripper_left = np.array([ep["gripper_left"] for ep in self.episode])
            gripper_right = np.array([ep["gripper_right"] for ep in self.episode])

            joint_left = np.array([ep["joint_left"] for ep in self.episode])
            joint_right = np.array([ep["joint_right"] for ep in self.episode])

            pose_left = np.array([ep["pose_left"] for ep in self.episode])
            pose_right = np.array([ep["pose_right"] for ep in self.episode])

            state_left = np.append(pose_left, gripper_left)
            state_right = np.append(pose_right, gripper_right)
            state_pose = np.append(state_right, state_left)

            state_left = np.append(joint_left, gripper_left)
            state_right = np.append(joint_right, gripper_right)
            state_joint = np.append(state_right, state_left)
        else:
            gripper = np.array([ep["gripper"] for ep in self.episode])
            joint = np.array([ep["joint"] for ep in self.episode])
            pose = np.array([ep["pose"] for ep in self.episode])

            state_joint = np.append(joint, gripper)
            state_pose = np.append(pose, gripper)
        return state_joint, state_pose
    
    def get_action(self):
        if self.condition["is_dual"]:
            gripper_left = np.array([ep["gripper_left"] for ep in self.episode])
            gripper_right = np.array([ep["gripper_right"] for ep in self.episode])
            action_left = np.array([ep["action_left"] for ep in self.episode])
            action_right = np.array([ep["action_right"] for ep in self.episode])
            action_left = np.append(action_left, gripper_left)
            action_right = np.append(action_right, gripper_right)

            action = np.append(action_right, action_left)
        else:
            gripper = np.array([ep["gripper"] for ep in self.episode])
            action = np.array([ep["action"] for ep in self.episode])
            action = np.append(action, gripper)
        return action

    def write(self, only_end=False):
        condition_path = os.path.join(self.condition["save_path"], f"{self.condition["task_name"]}.json")
        if not os.path.exists(condition_path):
            with open(condition_path, "w") as f:
                json.dump(self.condition, f)
        if only_end:
            save_path = os.path.join(self.condition["save_path"], f"{self.condition["task_name"]}/{self.episode_index}/")
            if os.path.exists(save_path):
                os.mkdir(save_path)
            npy_path = os.path.join(save_path, f"{len(self.episode)-1}.npy")
            data = self.episode[-1]
            np.save(npy_path, data)
            return
        # 直接存储为hdf5格式, 节省内存
        else:
            save_path = os.path.join(self.condition["save_path"], f"{self.condition["task_name"]}/{self.episode_index}.hdf5")
            if os.path.exists(save_path):
                os.mkdir(save_path)
            state_joint, state_pose = self.get_state()
            # 根据state_is_joint选择state
            if self.condition["state_is_joint"]:
                state = state_joint
            else:
                state = state_pose
            
            if self.condition["is_action"]:
                action = self.get_action()
            else:
                # action 为当前state, 注意,对于单步推理的模型不能这样设置!!!
                action = state

            enc_images_dict = self.encode_images()
                
            with h5py.File(save_path, "w") as f:
                f.create_dataset('action', data=np.array(action))
                obs = f.create_group('observations')
                image = obs.create_group('images')
                if self.condition["state_is_joint"]:
                    obs.create_dataset('qpos', data=state_joint)
                else:
                    obs.create_dataset('qpos', data=state_pose)
                # 存储着,保证信息没有丢失
                obs.create_dataset('state_pose', data=state_pose)
                obs.create_dataset('state_joint', data=state_joint)
                for key in enc_images_dict.keys():
                    image.create_dataset(key, data=enc_images_dict[key]["enc_data"], dtype=f'S{enc_images_dict[key]["len"]}')
                     
            # 清空当前的episode, 开始新的episode
            self.episode = []
            self.episode_index += 1
        return

def get_images(episode: list[dict], image_key: str) -> list[np.ndarray]:
    """
    从episode中获取所有图像
    """
    images = []
    if image_key not in episode[0].keys():
        print(f"image_key: {image_key} not in condition['image_keys']")
        return images

    for obs in episode:
        images.append(obs[image_key])
    return images

def images_encoding(imgs):
    encode_data = []
    max_len = 0

    # 单次循环完成编码和最大长度计算
    for img in imgs:
        success, encoded_image = cv2.imencode('.jpg', img)
        if not success:
            raise ValueError("Failed to encode image")
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))

    # 填充数据
    padded_data = [data.ljust(max_len, b'\0') for data in encode_data]

    return padded_data, max_len

    