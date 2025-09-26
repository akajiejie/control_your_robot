# HDF5数据格式转换器使用说明

## 概述

`prcesssed_hdf52openpi.py` 是一个用于将源HDF5文件转换为目标格式的转换器。它可以读取基于 `convert2act_hdf5.py` 格式的源数据，并将其转换为符合OpenPI或类似框架要求的目标格式。

## 功能特点

- **自动数据映射**: 自动将源数据格式映射到目标格式
- **图像处理**: 支持多相机图像数据的编码和调整大小
- **关节数据处理**: 处理单臂机器人的关节角度和夹爪状态
- **指令文件复制**: 自动复制源数据目录的`instructions.json`文件到每个episode文件夹
- **指令支持**: 支持自定义指令文件路径
- **错误处理**: 对损坏或不完整的数据文件进行跳过处理

## 数据格式映射

### 源格式 (基于convert2act_hdf5.py)
```
- slave_cam_head.color -> cam_high
- slave_cam_wrist.color -> cam_wrist  
- slave_left_arm.joint -> left_arm_joint
- slave_left_arm.gripper -> left_arm_gripper
```

### 目标格式
```
episode_X/
├── instructions.json
└── episode_X.hdf5
    ├── action (N-1, 7)
    └── observations/
        ├── qpos (N-1, 7)
        ├── left_arm_dim (N-1,)
        └── images/
            ├── cam_high (encoded JPEG)
            ├── cam_left_wrist (encoded JPEG)
            └── cam_right_wrist (encoded JPEG)
```

## 使用方法

### 命令行使用

```bash
# 基本用法
python scripts/prcesssed_hdf52openpi.py <source_path> <save_path>

# 指定处理的episode数量
python scripts/prcesssed_hdf52openpi.py <source_path> <save_path> --episode_num 50

# 完整示例
python scripts/prcesssed_hdf52openpi.py save/feed_rice/ processed_data/feed_rice_converted/ --episode_num 100
```

### 参数说明

- `source_path`: 源HDF5文件所在目录（转换器会自动在此目录中寻找instructions.json文件）
- `save_path`: 转换后数据的保存目录
- `--episode_num`: 可选，指定要处理的episode数量（默认处理所有）

### 编程使用

```python
from prcesssed_hdf52openpi import data_transform

# 转换数据（自动在源目录中寻找instructions.json）
success_count = data_transform(
    source_path="save/feed_rice/",
    save_path="processed_data/feed_rice_converted/",
    episode_num=50  # 可选
)

print(f"Successfully converted {success_count} episodes")
```

### 指令文件处理

转换器会自动在源数据目录中寻找`instructions.json`文件，并将其复制到每个episode文件夹中：

#### 输入格式

在源数据目录根目录下放置`instructions.json`文件（任何格式都可以）：

```json
{
    "instructions": [
        "Open the white drawer, grasp the blue cup, place it inside, and close the drawer.",
        "Pull open the white drawer, grip the blue cup, put it into the drawer, and shut it.",
        "Slide out the white drawer, pick up the blue cup, move it into the drawer, and push the drawer closed.",
        ...
    ]
}
```

或者其他格式：
```json
{
    "task_description": "Pick and place task",
    "instructions": "Complete the manipulation task",
    "metadata": {...}
}
```

#### 输出结果

每个episode文件夹都会包含完整的`instructions.json`文件副本：

```
processed_data/
└── task_converted/
    ├── episode_0/
    │   ├── instructions.json  ← 完整复制的文件
    │   └── episode_0.hdf5
    ├── episode_1/
    │   ├── instructions.json  ← 完整复制的文件
    │   └── episode_1.hdf5
    └── ...
```

- 转换器会自动在源数据目录中寻找`instructions.json`文件
- 直接复制整个`instructions.json`文件，不进行任何修改
- 支持任意格式的instructions.json文件
- 如果没有找到instructions.json，会创建默认的指令文件

## 测试

运行测试脚本来验证转换器功能：

```bash
python scripts/test_converter.py
```

测试脚本会：
1. 测试指令文件检测功能
2. 测试源数据读取功能
3. 执行小规模转换测试
4. 验证输出文件结构

## 注意事项

1. **数据完整性**: 转换器会跳过损坏或不完整的源文件
2. **图像尺寸**: 所有图像会被调整为640x480分辨率
3. **关节维度**: 单臂机器人数据（6个关节 + 1个夹爪 = 7维）
4. **内存使用**: 对于大量数据，建议分批处理

## 输出文件结构

转换后的数据结构如下：

```
processed_data/
└── task_name_converted/
    ├── episode_0/
    │   ├── instructions.json
    │   └── episode_0.hdf5
    ├── episode_1/
    │   ├── instructions.json
    │   └── episode_1.hdf5
    └── ...
```

每个episode包含：
- `instructions.json`: 任务指令
- `episode_X.hdf5`: 包含动作、观察和图像数据的HDF5文件
  - action: 7维动作序列（6个关节 + 1个夹爪）
  - qpos: 7维状态序列（6个关节 + 1个夹爪）
  - left_arm_dim: 左臂关节维度信息

## 故障排除

### 常见问题

1. **"No HDF5 files found"**: 检查源路径是否正确
2. **"Error reading HDF5"**: 源文件可能损坏，会自动跳过
3. **"No valid arm data found"**: 源文件中缺少关节数据

### 调试建议

1. 使用测试脚本验证数据读取
2. 检查源数据的键名是否与映射匹配
3. 确保有足够的磁盘空间保存转换结果

## 扩展

如果需要适配不同的数据格式，可以修改：

1. `source_map` 字典中的键值映射
2. `load_source_hdf5()` 函数中的数据读取逻辑
3. `convert_to_target_format()` 函数中的输出格式
