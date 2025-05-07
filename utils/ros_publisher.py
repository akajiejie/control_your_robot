import rospy
from geometry_msgs.msg import Twist
import threading

class ROSPublisher:
    def __init__(self, topic_name, msg_type):
        """
        初始化 ROS 发布者
        :param topic_name: 发布的话题名称
        :param msg_type: 消息类型
        """
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.publisher = None
        self.pub_msg = None
        self.shutdown_flag = False

        # 创建发布器
        self.publisher = rospy.Publisher(self.topic_name, self.msg_type, queue_size=10)

    def publish(self, event=None):
        """
        发布数据到指定的主题
        """
        if self.pub_msg is None:
            # rospy.logwarn("No message to publish.")
        if self.shutdown_flag:
            return
        else:
            self.publisher.publish(self.pub_msg)

    def continuous_publish(self):
        """
        使用定时器持续发布消息
        """
        # 每0.1秒调用一次publish函数
        rospy.Timer(rospy.Duration(0.1), self.publish)

    def update_msg(self, msg):
        """
        动态更新消息
        :param msg: Twist 消息实例
        """
        self.pub_msg = msg
    
    def stop(self):
        """
        停止发布
        """
        self.shutdown_flag = True
        rospy.loginfo("Publisher stopped.")

def start_publishing(publisher):
    """
    启动并执行连续发布的线程
    """
    publisher.continuous_publish()

    # 确保在ROS停止时清理资源
    rospy.on_shutdown(publisher.stop)

    # rospy.spin()

if __name__ == "__main__":
    try:
        # 创建 ROS 发布者实例
        publisher = ROSPublisher('/cmd_vel', Twist)
        # 初始化 ROS 节点
        rospy.init_node('ros_publisher_node', anonymous=True)
        
        # 初始化发布的消息
        msg = Twist()
        msg.linear.x = 0.1  # 设置线性速度
        publisher.update_msg(msg)

        # 启动一个独立的线程来执行连续发布
        pub_thread = threading.Thread(target=start_publishing, args=(publisher,))
        pub_thread.start()

        # 这里模拟一些操作，动态改变消息
        rospy.sleep(1)  # 保持停滞1秒
        msg.linear.x = 0.0
        publisher.update_msg(msg)

        rospy.sleep(1)  # 保持停滞1秒
        msg.linear.x = 0.1
        publisher.update_msg(msg)

        rospy.sleep(1)  # 保持停滞1秒
        msg.linear.x = 0.0
        publisher.update_msg(msg)

        # 停止发布消息
        rospy.sleep(1)  # 保持停滞1秒
        publisher.stop()
        rospy.loginfo("Shutting down ROS publisher.")

        # 等待线程退出
        pub_thread.join()

    except rospy.ROSInterruptException:
        pass
