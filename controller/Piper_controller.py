import sys
sys.path.append("./")

from controller.arm_controller import ArmController

from piper_sdk import *
import numpy as np
import time

class PiperController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None
    
    def set_up(self, can:str):
        piper = C_PiperInterface_V2(can)
        piper.ConnectPort()
        piper.EnableArm(7)
        enable_fun(piper=piper)
        self.controller = piper

    def reset(self, start_state):
        # 调用set_position或set_joint就行
        pass

    # 返回单位为米
    def get_state(self):
        state = {}
        eef = self.controller.GetArmEndPoseMsgs()
        joint = self.controller.GetArmJointMsgs()
        # 获取对应信息
        state["joint"] = np.array([joint.joint_state.joint_1, joint.joint_state.joint_2, joint.joint_state.joint_3,\
                                   joint.joint_state.joint_4, joint.joint_state.joint_5, joint.joint_state.joint_6]) * 0.001 / 180 * 3.1415926
        state["qpos"] = np.array([eef.end_pose.X_axis, eef.end_pose.Y_axis, eef.end_pose.Z_axis, \
                                  eef.end_pose.RX_axis, eef.end_pose.RY_axis, eef.end_pose.RZ_axis]) * 0.001 / 1000
        state["gripper"] = self.controller.GetArmGripperMsgs().gripper_state.grippers_angle * 0.001 / 70
        return state

    # 单位为米
    def set_position(self, position):
        x, y, z, rx, ry, rz = position*1000*1000
        x, y, z, rx, ry, rz = int(x), int(y), int(z), int(rx), int(ry), int(rz)

        self.controller.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.controller.EndPoseCtrl(x, y, z, rx, ry, rz)
    
    def set_joint(self, joint):
        j1, j2, j3 ,j4, j5, j6 = joint * 57295.7795 #1000*180/3.1415926
        j1, j2, j3 ,j4, j5, j6 = int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)
        self.controller.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.controller.JointCtrl(j1, j2, j3, j4, j5, j6)

    # 输入的是0~1的张合度
    def set_gripper(self, gripper):
        gripper = int(gripper * 70 * 1000)
        self.controller.GripperCtrl(gripper, 1000, 0x01, 0)

    def __del__(self):
        try:
            if hasattr(self, 'controller'):
                # Add any necessary cleanup for the arm controller
                pass
        except:
            pass

def enable_fun(piper:C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)

if __name__=="__main__":
    controller = PiperController("test_piper")
    controller.set_up("can0")
    print(controller.get_state())
    print(controller.get_gripper())

    controller.set_gripper(0.2)

    controller.set_joint(np.array([0.1,0.1,-0.2,0.3,-0.2,0.5]))
    time.sleep(1)
    print(controller.get_gripper())
    print(controller.get_state())

    controller.set_position(np.array([0.057, 0.0, 0.260, 0.0, 0.085, 0.0]))
    time.sleep(1)
    print(controller.get_gripper())
    print(controller.get_state())