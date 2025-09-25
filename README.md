[![中文](https://img.shields.io/badge/中文-简体-blue)](./README.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README_EN.md)

[Chinese WIKI](https://tian-nian.github.io/control_your_robot-doc/)

# Control Your Robot!
This project aims to provide a comprehensive and ready-to-use pipeline for embodied intelligence research, covering everything from robotic arm control, data collection, to Vision-Language-Action (VLA) model training and deployment.

## Quick Start!
Since this project includes several test examples, such as robotic arm tests, visual simulation, and full robot simulation, it is possible to quickly understand the overall framework without requiring any physical hardware.  
Because no hardware is needed, you can install the environment simply by running:

```
 pip install -r requirements.txt
```  
This project provides special debug levels: `"DEBUG"`, `"INFO"`, and `"ERROR"`. To fully observe the data flow, set it to `"DEBUG"`:
```bash
export INFO_LEVEL="DEBUG"
```

Alternatively, you can set it in the main function:
```python
import os
os.environ["INFO_LEVEL"] = "DEBUG" # DEBUG , INFO, ERROR
```

1. Data Collection Tests
```bash
# Multi-process (strict time-synchronized collection using TimeScheduler)
python example/collect/collect_mp_robot.py
# Multi-process (separate process for each component)
python example/collect/collect_mp_component.py
# Single-threaded (may have accumulated delays due to function execution)
python example/collect/collect.py
```

2. Model Deployment Tests
```bash
# Run a straightforward deployment test
python example/deploy/robot_on_test.py
# General deployment script
bash deploy.sh
# Offline data replay consistency test
bash eval_offline.sh
```

3. Remote Deployment and Data Transfer
```bash
# Start the server first, simulating the inference side (allows multiple connections, listens on a port)
python scripts/server.py
# On the client side, collect data and execute commands (example only executes 10 times)
python scripts/client.py
```

4. Interesting Scripts
```python
# Collect keypoints and perform trajectory replay
python scripts/collect_moving_ckpt.py 
# SAPIEN simulation, see planner/README.md for details
```

5. Debug Scripts
```bash
# Because controller and sensor packages have __init__.py, execute with -m
python -m controller.TestArm_controller
python -m sensor.TestVision_sensor
python -m my_robot.test_robot
```

6. Data Conversion Scripts
```bash
# After running python example/collect/collect.py and obtaining trajectories
python scripts/convert2rdt_hdf5.py save/test_robot/ save/rdt/
```

### 🤖 Supported Devices

#### 🎛️ Controllers
**✅ Implemented**
| Robotic Arm       | Mobile Base        | Dexterous Hand  | Others     |
|------------------|------------------|----------------|------------|
| Agilex Piper     | Agilex Tracer2.0 | 🚧 In Development | 📦 To Be Added |
| RealMan 65B      | Agilex bunker     | 📦 To Be Added    | 📦 To Be Added |
| Daran ALOHA      | 📦 To Be Added     | 📦 To Be Added    | 📦 To Be Added |
| Y1 ALOHA      | 📦 To Be Added     | 📦 To Be Added    | 📦 To Be Added |

**🚧 Planned Support**
| Robotic Arm      | Mobile Base       | Dexterous Hand | Others     |
|-----------------|-----------------|----------------|------------|
| JAKA             | 📦 To Be Added    | 📦 To Be Added | 📦 To Be Added |
| Franka           | 📦 To Be Added    | 📦 To Be Added | 📦 To Be Added |
| UR5e             | 📦 To Be Added    | 📦 To Be Added | 📦 To Be Added |

#### 📡 Sensors
**✅ Implemented**
| Vision Sensors   | Tactile Sensors | Other Sensors |
|-----------------|----------------|---------------|
| RealSense Series | Vitac3D        | 📦 To Be Added |

**🚧 Planned Support**
For new sensor support requests, please open an issue, or submit a PR with your sensor configuration!

## Directory Overview
| Directory       | Description                  | Main Content |
|----------------|-----------------------------|--------------|
| **📂 controller** | Robot controller wrappers  | Classes for controlling arms, mobile bases, etc. |
| **📂 sensor**    | Sensor wrappers            | Currently only RealSense cameras |
| **📂 utils**     | Utility functions          | Math, logging, and other helper functions |
| **📂 data**      | Data collection module     | Classes for data recording and processing |
| **📂 my_robot**  | Robot integration wrappers | Full robot system composition classes |
| **📂 policy**    | VLA model policies         | Vision-Language-Action model implementations |
| **📂 scripts**   | Example scripts            | Main entry points and test scripts |
| **📂 third_party** | Third-party dependencies | External libraries requiring compilation |
| **📂 planner**   | Motion planning module     | `curobo` planner wrappers + simulated robot code |
| **📂 example**   | Example workflows          | Data collection, model deployment examples |
