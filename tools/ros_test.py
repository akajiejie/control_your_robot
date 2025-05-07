import rospy
from geometry_msgs.msg import Twist

def move_robot():
    # 初始化 ROS 节点
    rospy.init_node('robot_controller', anonymous=True)
    
    # 创建发布器，发布到 /cmd_vel 话题
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # 创建 Twist 消息实例
    move_cmd = Twist()
    
    # 设置线性速度和角速度
    move_cmd.linear.x = 0.5  # 直线速度 0.5 m/s，表示前进
    move_cmd.angular.z = 0.0  # 角速度 0.2 rad/s，表示旋转

    # 设置发布频率
    rate = rospy.Rate(10)  # 设置为 10Hz（每秒发布 10 次）

    # 发布前进指令并持续控制
    rospy.loginfo("Robot moving...")
    for _ in range(50):  # 发布 50 次，相当于持续前进 5 秒钟
        pub.publish(move_cmd)
        rate.sleep()  # 控制发布频率为 10Hz

    # 停止机器人
    rospy.loginfo("Robot stopped.")
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 0.0
    pub.publish(move_cmd)  # 发布停止消息

if __name__ == "__main__":
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass