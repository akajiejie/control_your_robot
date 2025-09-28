---
license: apache-2.0
---

## Latest Updates
- [2025-09-24] 🚀 v0.2 Dataset Released: 1015 episodes across five tasks. Available in both Arrow and LMDB formats. The v0.2 dataset was created after we re-calibrated the zero point of our robotic arm. 
- [2025-09-04] v0.1 Dataset Released: 1086 episodes across five tasks. Available in Arrow and LMDB formats. (See note on zero-point drift).

# 1. Data Introduction
### Data Format
This project provides robotic manipulation datasets in two formats: Arrow and LMDB:
- Arrow Dataset: Built on the [Apache Arrow](https://arrow.apache.org/) format. Its column-oriented structure offers flexibility and will be the primary format for development in robo_orchard_lab. It features standardized message types and supports exporting to Mcap files for visualization.
- LMDB Dataset: Built on the [LMDB](https://github.com/LMDB) (Lightning Memory-Mapped Database) format, which is optimized for extremely fast read speeds.

### &#9888; Important Note on Dataset Versions

The v0.1 dataset was affected by a robotic arm zero-point drift issue during data acquisition. We have since re-calibrated the arm and collected the v0.2 dataset.
- v0.2: Please use this version for all fine-tuning and evaluation to ensure model accuracy.
- v0.1: This version should only be used for pre-training experiments or deprecated entirely.

### Verifying Hardware Consistency

If you are using your own Piper robot arm, you can check for the same zero-point drift issue:

1. Check Hardware Zero Alignment: Home the robot arm and visually inspect if each joint aligns correctly with the physical zero-point markers.
2. Replay v0.2 Dataset: Replay the joint states from the v0.2 dataset. If the arm successfully completes the tasks, your hardware setup is consistent with ours.

## 1.1 Version 0.2
| Task    | Episode Num | LMDB Dataset | Arrow Dataset |
| :--------: | :-------: |:-------: | :-------: |
| place_shoe  | 220    |  lmdb_dataset_place_shoe_2025_09_11 | arrow_dataset_place_shoe_2025_09_11
| empty_cup_place | 196 | lmdb_dataset_empty_cup_place_2025_09_09 | arrow_dataset_empty_cup_place_2025_09_09 | 
| put_bottles_dustbin    | 199  | lmdb_dataset_put_bottles_dustbin_2025_09_11  | lmdb_dataset_put_bottles_dustbin_2025_09_11
| stack_bowls_three    | 200    | lmdb_dataset_stack_bowls_three_2025_09_09<br>lmdb_dataset_stack_bowls_three_2025_09_10  |arrow_dataset_stack_bowls_three_2025_09_09<br>arrow_dataset_stack_bowls_three_2025_09_10
| stack_blocks_three    | 200  | lmdb_dataset_stack_blocks_three_2025_09_10 |  arrow_dataset_stack_blocks_three_2025_09_10 |


## 1.2 Version 0.1
| Task    | Episode Num | LMDB Dataset | Arrow Dataset |
| :--------: | :-------: |:-------: | :-------: |
| place_shoe  | 200    |  lmdb_dataset_place_shoe_2025_08_21<br>lmdb_dataset_place_shoe_2025_08_27 | arrow_dataset_place_shoe_2025_08_21<br>arrow_dataset_place_shoe_2025_08_27
| empty_cup_place | 200 | lmdb_dataset_empty_cup_place_2025_08_19 | arrow_dataset_empty_cup_place_2025_08_19 | 
| put_bottles_dustbin    | 200  | lmdb_dataset_put_bottles_dustbin_2025_08_20<br>lmdb_dataset_put_bottles_dustbin_2025_08_21  | arrow_dataset_put_bottles_dustbin_2025_08_20<br>arrow_dataset_put_bottles_dustbin_2025_08_21
| stack_bowls_three    | 219    | lmdb_dataset_stack_bowls_three_2025_08_19<br>lmdb_dataset_stack_bowls_three_2025_08_20  |arrow_dataset_stack_bowls_three_2025_08_19<br>arrow_dataset_stack_bowls_three_2025_08_20
| stack_blocks_three    | 267  | lmdb_dataset_stack_blocks_three_2025_08_26<br>lmdb_dataset_stack_blocks_three_2025_08_27 |  arrow_dataset_stack_blocks_three_2025_08_26<br>arrow_dataset_stack_blocks_three_2025_08_27 |

# 2. Usage Example

## 2.1 LMDB Dataset Usage Example

Ref to [RoboTwinLmdbDataset](https://github.com/HorizonRobotics/robo_orchard_lab/blob/master/robo_orchard_lab/dataset/robotwin/robotwin_lmdb_dataset.py) class from robo_orchard_lab. See [SEM config](https://github.com/HorizonRobotics/robo_orchard_lab/blob/master/projects/sem/robotwin/config_sem_robotwin.py#L42) for a usage example. 


## 2.2 Arrow Dataset Usage Example

Ref to [ManipulationRODataset](https://github.com/HorizonRobotics/robo_orchard_lab/blob/master/robo_orchard_lab/dataset/robotwin/arrow_dataset.py) class from robo_orchard_lab. Here is some usage example:

### 2.2.1 Data Parse Example
```python
def build_dataset(config):
    from robo_orchard_lab.dataset.robot.dataset import (
        ROMultiRowDataset,
        ConcatRODataset,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import ArrowDataParse
    from robo_orchard_lab.dataset.robotwin.transforms import EpisodeSamplerConfig
    
    dataset_list = []
    data_parser = ArrowDataParse(
        cam_names=config["cam_names"],
        load_image=True,
        load_depth=True,
        load_extrinsic=True,
        depth_scale=1000,
    )
    joint_sampler = EpisodeSamplerConfig(target_columns=["joints", "actions"])

    for path in config["data_path"]:
        dataset = ROMultiRowDataset(
            dataset_path=path, row_sampler=joint_sampler
        )
        dataset.set_transform(data_parser)
        dataset_list.append(dataset)

    dataset = ConcatRODataset(dataset_list)
    return dataset

config = dict(
    data_path=[
        "data/arrow_dataset_place_shoe_2025_08_21",
        "data/arrow_dataset_place_shoe_2025_08_27",
    ],
    cam_names=["left", "middle", "right"],
)
dataset = build_dataset(config)

# Show all key
frame_index = 0
print(len(dataset))
print(dataset[frame_index].keys())

# Show important key
for key in ['joint_state', 'master_joint_state', 'imgs', 'depths', 'intrinsic', 'T_world2cam']:
    print(f"{key}, shape is {dataset[frame_index][key].shape}")
print(f"Instuction: {dataset[frame_index]['text']}")
print(f"Dataset index: {dataset[frame_index]['dataset_index']}")

# ----Output Demo----
# joint_state, shape is (322, 14)
# master_joint_state, shape is (322, 14)
# imgs, shape is (3, 360, 640, 3)
# depths, shape is (3, 360, 640)
# intrinsic, shape is (3, 4, 4)
# T_world2cam, shape is (3, 4, 4)
# Instuction: Use one arm to grab the shoe from the table and place it on the mat.
# Dataset index: 1

```

### 2.2.2 For Training
To integrate this dataset into the training pipeline, you will need to incorporate data transformations. Please follow the approach used in the [lmdb_dataset](https://github.com/HorizonRobotics/RoboOrchardLab/blob/master/projects/sem/robotwin/config_sem_robotwin.py) to add the transforms.

```python
from robo_orchard_lab.dataset.robotwin.transforms import ArrowDataParse
from robo_orchard_lab.utils.build import build
from robo_orchard_lab.utils.misc import as_sequence    
from torchvision.transforms import Compose 
train_transforms, val_transforms = build_transforms(config)
train_transforms = [build(x) for x in as_sequence(train_transforms)]

composed_train_transforms = Compose([data_parser] + train_transforms)
train_dataset.set_transform(composed_train_transforms)
```
### 2.2.3 Export mcap file and use foxglove to viz
```
def export_mcap(dataset, episode_index, target_path):
    """Export the specified episode to an MCAP file."""
    from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.camera import (  # noqa: E501
        McapBatchFromBatchCameraDataEncodedConfig,
    )
    from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.joint_state import (  # noqa: E501
        McapBatchFromBatchJointStateConfig,
    )
    from robo_orchard_lab.dataset.experimental.mcap.writer import (
        Dataset2Mcap,
        McapBatchEncoderConfig,
    )

    dataset2mcap_cfg: dict[str, McapBatchEncoderConfig] = {
        "joints": McapBatchFromBatchJointStateConfig(
            target_topic="/observation/robot_state/joints"
        ),
    }
    dataset2mcap_cfg["actions"] = McapBatchFromBatchJointStateConfig(
        target_topic="/action/robot_state/joints"
    )

    for camera_name in config["cam_names"]:
        dataset2mcap_cfg[camera_name] = (
            McapBatchFromBatchCameraDataEncodedConfig(
                calib_topic=f"/observation/cameras/{camera_name}/calib",
                image_topic=f"/observation/cameras/{camera_name}/image",
                tf_topic=f"/observation/cameras/{camera_name}/tf",
            )
        )
        dataset2mcap_cfg[f"{camera_name}_depth"] = (
            McapBatchFromBatchCameraDataEncodedConfig(
                image_topic=f"/observation/cameras/{camera_name}/depth",
            )
        )

    to_mcap = Dataset2Mcap(dataset=dataset)
    to_mcap.save_episode(
        target_path=target_path,
        episode_index=episode_index,
        encoder_cfg=dataset2mcap_cfg,
    )
    print(f"Export episode {episode_index} to {target_path}")


# Export mcap file and use foxglove to viz
dataset_index = dataset[frame_index]["dataset_index"]
episode_index = dataset[frame_index]["episode"].index
export_mcap(
    dataset=dataset.datasets[dataset_index],
    episode_index=episode_index,
    target_path=f"./viz_dataidx_{dataset_index}_episodeidx_{episode_index}.mcap",
)
```

Then you can use [Foxglove](https://foxglove.dev/) and [Example Layout](./arrow_foxglove_layout.json) to visualize the mcap file. Refer to [here](https://foxglove.dev/examples) to get more visualization example.
