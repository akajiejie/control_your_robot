import sys
sys.path.append('./')

# from my_robot.test_robot import TestRobot
from utils.data_handler import hdf5_groups_to_dict
from my_robot.base_robot import dict_to_list
from data.collect_any import CollectAny
from policy.DP.inference_model import MYDP
import time
import numpy as np
import math
import cv2
import h5py
import os

condition = {
    "save_path": "./test/reload_model_actions/", 
    "task_name": "dp_feed_test", 
    "save_format": "hdf5", 
    "save_freq": 10,
    "collect_type": "reload",
}
def decode_compressed_images(encoded_data):
    """Decode compressed JPEG image data
    
    Args:
        encoded_data: Compressed JPEG byte data array
        
    Returns:
        Decoded image array
    """
    imgs = []
    for data in encoded_data:
        # Remove padded zero bytes
        jpeg_bytes = data.tobytes().rstrip(b"\0")
        # Convert byte data to numpy array
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        # Decode JPEG image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
        else:
            print("Warning: Unable to decode image data")
    return np.array(imgs)

class Replay:
    def __init__(self, hdf5_path, is_compressed=False) -> None:
        self.ptr = 0
        self.is_compressed = is_compressed
        
        if is_compressed:
            # Handle compressed format data
            self.episode_data = self._load_compressed_data(hdf5_path)
        else:
            # Original format
            self.episode = dict_to_list(hdf5_groups_to_dict(hdf5_path))
    
    def _load_compressed_data(self, hdf5_path):
        """Load HDF5 data (supports multiple formats)"""
        episode_data = []
        
        with h5py.File(hdf5_path, 'r') as f:
            # Detect HDF5 file structure
            top_level_keys = list(f.keys())
            print(f"Detected top-level keys: {top_level_keys}")
            
            # Determine data format
            if 'observations' in top_level_keys:
                # Format 1: Compressed format with observations layer
                obs = f['observations']
                
                # Decode compressed image data - supports three cameras
                cam_high_encoded = obs['cam_high'][:]
                cam_high_images = decode_compressed_images(cam_high_encoded)
                
                # Check for left and right wrist cameras
                cam_left_wrist_images = None
                cam_right_wrist_images = None
                cam_wrist_images = None
                
                if 'cam_left_wrist' in obs and 'cam_right_wrist' in obs:
                    # Dual-arm configuration: left and right wrist cameras
                    cam_left_wrist_encoded = obs['cam_left_wrist'][:]
                    cam_right_wrist_encoded = obs['cam_right_wrist'][:]
                    cam_left_wrist_images = decode_compressed_images(cam_left_wrist_encoded)
                    cam_right_wrist_images = decode_compressed_images(cam_right_wrist_encoded)
                elif 'cam_wrist' in obs:
                    # Single-arm configuration: single wrist camera
                    cam_wrist_encoded = obs['cam_wrist'][:]
                    cam_wrist_images = decode_compressed_images(cam_wrist_encoded)
                
                # Get robot arm data
                left_arm_joint = obs['left_arm']['joint'][:]
                left_arm_gripper = obs['left_arm']['gripper'][:]
                
                # Check for right arm data
                right_arm_joint = None
                right_arm_gripper = None
                if 'right_arm' in obs and len(obs['right_arm'].keys()) > 0:
                    if 'joint' in obs['right_arm'] and 'gripper' in obs['right_arm']:
                        right_arm_joint = obs['right_arm']['joint'][:]
                        right_arm_gripper = obs['right_arm']['gripper'][:]
                
            else:
                # Format 2: Directly stored uncompressed format (slave_cam_head, slave_left_arm, etc.)
                # Read image data (already in decoded format)
                if 'slave_cam_head' in f:
                    cam_high_images = f['slave_cam_head']['color'][:]
                elif 'cam_high' in f:
                    cam_high_images = f['cam_high']['color'][:]
                else:
                    raise KeyError("Head camera data not found (slave_cam_head or cam_high)")
                
                # Check wrist cameras
                cam_left_wrist_images = None
                cam_right_wrist_images = None
                cam_wrist_images = None
                
                if 'slave_cam_wrist' in f:
                    cam_wrist_images = f['slave_cam_wrist']['color'][:]
                elif 'cam_wrist' in f:
                    cam_wrist_images = f['cam_wrist']['color'][:]
                elif 'cam_left_wrist' in f and 'cam_right_wrist' in f:
                    cam_left_wrist_images = f['cam_left_wrist']['color'][:]
                    cam_right_wrist_images = f['cam_right_wrist']['color'][:]
                
                # Read robot arm data
                if 'slave_left_arm' in f:
                    left_arm_joint = f['slave_left_arm']['joint'][:]
                    left_arm_gripper = f['slave_left_arm']['gripper'][:]
                elif 'left_arm' in f:
                    left_arm_joint = f['left_arm']['joint'][:]
                    left_arm_gripper = f['left_arm']['gripper'][:]
                else:
                    raise KeyError("Left arm data not found (slave_left_arm or left_arm)")
                
                # Check right arm data
                right_arm_joint = None
                right_arm_gripper = None
                if 'slave_right_arm' in f:
                    right_arm_joint = f['slave_right_arm']['joint'][:]
                    right_arm_gripper = f['slave_right_arm']['gripper'][:]
                elif 'right_arm' in f and len(f['right_arm'].keys()) > 0:
                    if 'joint' in f['right_arm'] and 'gripper' in f['right_arm']:
                        right_arm_joint = f['right_arm']['joint'][:]
                        right_arm_gripper = f['right_arm']['gripper'][:]
            
            # Build episode data
            for i in range(len(left_arm_joint)):
                step_data = {
                    'left_arm': {
                        'joint': left_arm_joint[i],
                        'gripper': left_arm_gripper[i]
                    },
                    'cam_high': {
                        'color': cam_high_images[i] if i < len(cam_high_images) else cam_high_images[-1]
                    }
                }
                
                # Add right arm data (if exists)
                if right_arm_joint is not None and right_arm_gripper is not None:
                    step_data['right_arm'] = {
                        'joint': right_arm_joint[i],
                        'gripper': right_arm_gripper[i]
                    }
                
                # Add camera data
                if cam_left_wrist_images is not None and cam_right_wrist_images is not None:
                    # Dual-arm configuration
                    step_data['cam_left_wrist'] = {
                        'color': cam_left_wrist_images[i] if i < len(cam_left_wrist_images) else cam_left_wrist_images[-1]
                    }
                    step_data['cam_right_wrist'] = {
                        'color': cam_right_wrist_images[i] if i < len(cam_right_wrist_images) else cam_right_wrist_images[-1]
                    }
                elif cam_wrist_images is not None:
                    # Single-arm configuration
                    step_data['cam_wrist'] = {
                        'color': cam_wrist_images[i] if i < len(cam_wrist_images) else cam_wrist_images[-1]
                    }
                
                episode_data.append(step_data)
        
        return episode_data
    
    def get_data(self):
        if self.is_compressed:
            # Compressed format: return current step data
            if self.ptr >= len(self.episode_data):
                return None, None
            
            step_data = self.episode_data[self.ptr]
            
            # Build robot arm data
            arm_data = {
                'left_arm': step_data['left_arm']
            }
            
            # Add right arm data (if exists)
            if 'right_arm' in step_data:
                arm_data['right_arm'] = step_data['right_arm']
            
            # Build image data
            img_data = {
                'cam_high': step_data['cam_high']
            }
            
            # Add camera data (based on actually existing cameras)
            if 'cam_left_wrist' in step_data and 'cam_right_wrist' in step_data:
                # Dual-arm configuration
                img_data['cam_left_wrist'] = step_data['cam_left_wrist']
                img_data['cam_right_wrist'] = step_data['cam_right_wrist']
            elif 'cam_wrist' in step_data:
                # Single-arm configuration
                img_data['cam_wrist'] = step_data['cam_wrist']
            #horizon - (n_obs_steps - 1)= 8 - (3 - 1)=6
            self.ptr += 6
            return arm_data, img_data
        else:
            # Original format
            data = self.episode[self.ptr], self.episode[self.ptr]
            self.ptr += 6
            return data

    
def input_transform(data):
    has_left_arm = "left_arm" in data[0]
    has_right_arm = "right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": 0.0
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["left_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": 0.0
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
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    ])
    
    # Process image data - support different camera configurations
    if "cam_left_wrist" in data[1] and "cam_right_wrist" in data[1]:
        # Dual-arm configuration: three cameras
        img_arr = (
            data[1]["cam_high"]["color"], 
            data[1]["cam_left_wrist"]["color"],
            data[1]["cam_right_wrist"]["color"]
        )
    elif "cam_wrist" in data[1]:
        # Single-arm configuration: two cameras
        img_arr = (
            data[1]["cam_high"]["color"], 
            data[1]["cam_wrist"]["color"]
        )
    else:
        # Only head camera
        img_arr = (data[1]["cam_high"]["color"],)
    
    return img_arr, state

def output_transform(data):
    joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),    # joint2
        (math.radians(-170), math.radians(0)),   # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),   # joint5
        (math.radians(-120), math.radians(120))    # joint6
        ]
    def clamp(value, min_val, max_val):
        """Clamp value to [min_val, max_val] range"""
        return max(min_val, min(value, max_val))
    left_joints = [
        clamp(data[0][i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    right_joints = [
        clamp(data[0][i+7], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    if data[0][6] < 0.05:
        data[0][6] = 0.0
    left_gripper = data[0][6]
    if data[0][13] < 0.05:
        data[0][13] = 0.0
    right_gripper = data[0][13]
    move_data = {
        "left_arm":{
            "joint": left_joints,
            "gripper": left_gripper,
        },
        "right_arm":{
            "joint": right_joints,
            "gripper": right_gripper,
        }

    }
    return move_data

if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    model = MYDP(model_path="policy/DP/checkpoints/feed_test-100-0/300.ckpt", task_name="feed_test", INFO="DEBUG")
    collection=CollectAny(condition=condition,start_episode=0,move_check=True,resume=False)
    time.sleep(1)

    # Source folder path
    source_folder = "save/real_data/feed_test/"
    
    # Get all hdf5 files in folder
    hdf5_files = []
    for filename in os.listdir(source_folder):
        if filename.endswith('.hdf5'):
            hdf5_files.append(os.path.join(source_folder, filename))
    
    # Sort by numeric order in filename
    def extract_number(filename):
        """Extract numbers from filename for sorting"""
        import re
        basename = os.path.basename(filename)
        # Extract numeric part from filename
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[0])  # Use first number for sorting
        return 0  # If no number, return 0
    
    hdf5_files.sort(key=extract_number)
    
    if not hdf5_files:
        print(f"No hdf5 files found in folder {source_folder}")
        exit()
    
    print(f"Found {len(hdf5_files)} hdf5 files (sorted by numeric order):")
    for i, file in enumerate(hdf5_files):
        basename = os.path.basename(file)
        print(f"  {i+1}. {basename} (full path: {file})")
    
    # Iterate through each hdf5 file for testing
    for file_idx, source_hdf5_path in enumerate(hdf5_files):
        print(f"\nStarting to process file {file_idx + 1}/{len(hdf5_files)}: {source_hdf5_path}")
        
        re = Replay(source_hdf5_path, is_compressed=True)
        if re.is_compressed:
            max_step = len(re.episode_data)  # Compressed format uses episode_data length
        else:
            max_step = len(re.episode)  # Original format uses episode length
        print(f"HDF5 file contains {max_step} steps of data")
        
        # Extract filename from source file path (without extension)
        source_filename = os.path.splitext(os.path.basename(source_hdf5_path))[0]
        print(f"Source filename: {source_filename}")
        
        # Process current file
        step = 0
        # Reset all information
        model.reset_obsrvationwindows()
        model.random_set_language()
        
        time.sleep(1)

        # Start inference run step by step
        while step < max_step:
            raw_data=re.get_data()
            img_arr, state = input_transform(raw_data)
            
            model.update_observation_window(img_arr, state)
            action_chunk = model.get_action(model.observation_window) 
            # print(action_chunk)
            for action in action_chunk:
                # Convert action data to format expected by collect_any
                controllers_data = {
                    "left_arm": {
                        "joint": action[:6].tolist(),  # First 6 are left arm joint angles
                        "gripper": action[6]  # 7th is left arm gripper state
                    },
                    "right_arm": {
                        "joint": action[7:13].tolist(),  # 8th-13th are right arm joint angles
                        "gripper": action[13]  # 14th is right arm gripper state
                    }
                }
                collection.collect(controllers_data, None)
                step += 1
                # pdb.set_trace()
                time.sleep(1/condition["save_freq"])
                print(f"File {file_idx + 1}/{len(hdf5_files)}, Step {step}/{max_step} completed. Action saved.")

        collection.write()
        time.sleep(1)
        print(f"Completed processing file {source_filename}")
    
    print(f"\nAll files processed! Total processed {len(hdf5_files)} files")