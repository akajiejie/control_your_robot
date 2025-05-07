import rospy
import threading
from typing import Callable, Optional

class ROSSubscriber:
    def __init__(self, topic_name, msg_type,call: Optional[Callable] = None):
        """
        初始化 ROS 订阅器
        :param topic_name: 订阅的话题名称
        :param msg_type: 消息类型
        """
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.latest_msg = None
        self.lock = threading.Lock()
        self.user_call = call
        
        # 订阅话题
        self.subscriber = rospy.Subscriber(self.topic_name, self.msg_type, self.callback)
    
    def callback(self, msg):
        """
        订阅回调函数，用于接收消息并更新最新的数据。
        :param msg: 收到的消息
        """
        with self.lock:
            self.latest_msg = msg
            if self.user_call:
                self.user_call(self.latest_msg)

    def get_latest_data(self):
        """
        获取最新的订阅数据，确保数据为最新
        :return: 最新的数据（例如 PointCloud2 消息）
        """
        with self.lock:
            return self.latest_msg

        
if __name__=="__main__":
    import time
    from tracer_msgs.msg import TracerRsStatus

    ros_test = ROSSubscriber('/tracer_rs_status', TracerRsStatus)
    # 初始化 ROS 节点
    rospy.init_node('ros_subscriber_node', anonymous=True)
    for i in range(100):
        print(ros_test.get_latest_data())
        time.sleep(0.1)

