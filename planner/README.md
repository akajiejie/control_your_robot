### 通用的机械臂planner
由于好多厂家自带的planner很垃圾,所以我结合了RoboTwin2.0最新的planner,实现了一个通用接口.  
考虑到有些朋友还没有自己的机器人,所以顺便给出了RoboTwin中我们最喜欢的cobomagic双臂平台的操控示例啦.  
这个会慢慢做成一个仿真教程~

机械臂URDF:
通过网盘分享的文件：仿真机械臂
链接: https://pan.baidu.com/s/1Mfrs3spVTeRWUHf_pyHZjQ?pwd=yq7m  
提取码: yq7m 

需要修改`curobo_left.yml`和`curobo_right.yml`中`collision_spheres`和`urdf_path`,要求为绝对路径.

环境配置:
```bash
# 安装sapien基础环境
pip install - r requirements.txt

# 安装curobo
cd ../third_party
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..
```

### 已经实现
| 日期       | 更新内容                          | 状态     |
|------------|----------------------------------|----------|
| 2025.5.22 | 🤖仿真环境中的通用IK示例              | ✅ 已发布 |
| 2025.5.22 | 💻通用planner接入                   | ✅ 已发布 |
| 2025.5.22 | 🏙️接入D435仿真摄像头设置              | ✅ 已发布 |

### 正在路上
- [ ] 📷多种camera设置支持
- [ ] 📖URDF, sapien使用简单教学与示例

