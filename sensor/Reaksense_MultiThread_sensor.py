import numpy as np
import pyrealsense2 as rs
import time
from sensor.vision_sensor_mutithread import MutiThreadVisionSensor
from copy import deepcopy
import threading
from collections import deque
from utils.data_handler import debug_print
import cv2
import logging
logger = logging.getLogger(__name__)

def find_device_by_serial(devices, serial):
    """Find device index by serial number"""
    for i, dev in enumerate(devices):
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return i
    return None

class RealsenseSensor(MutiThreadVisionSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        # 线程控制
        self._exit_event = threading.Event()
        self._thread = None
        self._frame_lock = threading.Lock()
        self._frame_buffer = deque(maxlen=1)  # 仅保留最新帧
        self._new_frame_event = threading.Event()
        self._first_frame_event = threading.Event()
        
        # 后处理选项
        self.rotation = None  # 可以设置为cv2.ROTATE_90_CLOCKWISE等
        self.color_mode = "bgr"  # 或 "rgb"
        
        # RealSense对象
        self._config = None
        self._context = None
        self._profile = None
        
    def set_up(self,CAMERA_SERIAL,is_depth = False):
        self.is_depth = is_depth
        try:
            # Initialize RealSense context and check for connected devices
            self.context = rs.context()
            self.devices = list(self.context.query_devices())
            
            if not self.devices:
                raise RuntimeError("No RealSense devices found")
            
            # Initialize each camera
            serial = CAMERA_SERIAL
            device_idx = find_device_by_serial(self.devices, serial)
            if device_idx is None:
                raise RuntimeError(f"Could not find camera with serial number {serial}")
            
            self.pipeline = rs.pipeline()
            self.config = rs.config()
        
            # Enable device by serial number
            self.config.enable_device(serial)
            # self.config.disable_all_streams()
            # Enable color stream only
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            if is_depth:
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start streaming
            try:
                self.pipeline.start(self.config)

                #start mutitheading process
                if hasattr(self, 'collect_info') and self.collect_info:  # 显式检查非空
                    
                    self._start_thread()
                    self.wait_first_frame()  # 等待首帧就绪
                print(f"Started camera: {self.name} (SN: {serial})")
            except RuntimeError as e:
                raise RuntimeError(f"Error starting camera: {str(e)}")
        except Exception as e:
            self.cleanup_mp()
            raise RuntimeError(f"Failed to initialize camera: {str(e)}")
    def wait_first_frame(self, timeout=10):
        """阻塞等待首帧就绪"""
        if not self._first_frame_event.wait(timeout=timeout):
            raise TimeoutError(f"{self.name} 首帧等待超时")
        return True
    def _start_thread(self):
        """启动采集线程"""
        if self._thread and self._thread.is_alive():
            logger.warning(f"{self.name} 线程已在运行")
            return
            
        self._exit_event.clear()
        self._thread = threading.Thread(target=self._update_frames, name=f"{self.name}_thread")
        self._thread.daemon = True
        self._thread.start()
        logger.debug(f"已启动 {self.name} 的采集线程")
    def _update_frames(self):
        """独立线程持续获取帧数据"""
        try:
            while not self._exit_event.is_set():
                # 等待帧，超时时间设为1000ms
                frames = self.pipeline.wait_for_frames(1000)
                
                # 检查退出信号
                if self._exit_event.is_set():
                    break
                    
                frame_data = {}
                
                # 处理彩色帧
                if "color" in self.collect_info:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        frame_data["color"] = color_image[:,:,::-1]
                
                # 处理深度帧
                if self.is_depth and "depth" in self.collect_info:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_map = np.asanyarray(depth_frame.get_data())
                        frame_data["depth"] = depth_map
                
                # 如果有有效数据，更新缓冲区
                if frame_data:
                    with self._frame_lock:
                        self._frame_buffer.append(frame_data)
                    
                    # 设置新帧事件
                    self._new_frame_event.set()
                    
                    # 设置首帧事件（如果尚未设置）
                    if not self._first_frame_event.is_set():
                        self._first_frame_event.set()
        
        except RuntimeError as e:
            if "timeout" in str(e):
                logger.warning(f"{self.name} 帧等待超时")
            else:
                logger.error(f"{self.name} 捕获到RuntimeError: {str(e)}")
        except Exception as e:
            logger.error(f"{self.name} 捕获到异常: {str(e)}")
        finally:
            logger.info(f"{self.name} 采集线程退出")
        #
    def get_image_mp(self,timeout_ms=300):
        """获取最新帧（带超时等待）"""
        # 等待新帧到达
        if not self._new_frame_event.wait(timeout_ms / 1000.0):
            thread_alive = self._thread is not None and self._thread.is_alive()
            raise TimeoutError(
                f"等待 {self.name} 的新帧超时 ({timeout_ms}ms)。"
                f"采集线程状态: {'运行中' if thread_alive else '已停止'}"
            )
        
        # 重置事件
        self._new_frame_event.clear()
        
        # 获取最新帧
        with self._frame_lock:
            if self._frame_buffer:
                return deepcopy(self._frame_buffer[-1])
            return None
    def cleanup(self):
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    def cleanup_mp(self):
        """清理资源"""
        # 设置退出事件
        self._exit_event.set()
        
        # 等待线程退出
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(f"{self.name} 线程未正常退出")
        
        # 停止pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
                logger.info(f"{self.name} pipeline已停止")
            except Exception as e:
                logger.error(f"停止 {self.name} pipeline时出错: {str(e)}")
            finally:
                self.pipeline = None
        
        # 重置状态
        self._frame_buffer.clear()
        self._new_frame_event.clear()
        self._first_frame_event.clear()
        self._thread = None
        self._profile = None
        
        logger.info(f"{self.name} 资源已清理")
    def __del__(self):
        self.cleanup_mp()

if __name__ == "__main__":
    cam = RealsenseSensor("head")
    cam1= RealsenseSensor("left")
    # cam2= RealsenseSensor("right")

    cam.set_up("420122070816")
    cam1.set_up("948122073452")
    # cam2.set_up("338622074268")
    
    cam.set_collect_info(["color"])
    cam1.set_collect_info(["color"])
    cam._start_thread()
    cam1._start_thread()
    # cam2.set_collect_info(["color"])
    cam.wait_first_frame()
    cam1.wait_first_frame()
    cam_list = []
    cam_list1 = []
    # cam_list2 = []

    for i in range(300):
        print(i)
        data = cam.get_image_mp()
        data1 = cam1.get_image_mp()
        # data2 = cam2.get_image_mp()
        
        cam_list.append(data)
        cam_list1.append(data1)
        # cam_list2.append(data2)
        
        time.sleep(0.1)
