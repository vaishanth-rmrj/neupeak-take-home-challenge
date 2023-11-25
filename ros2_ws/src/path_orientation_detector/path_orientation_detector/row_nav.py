import numpy as np
import time
from .orientation_detector import PathOrientationDetector

# ros2 imports
import rclpy
from rclpy.node import Node
import pkg_resources
# ros msg imports
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker

# external imports
from .utils import generate_pointcloud2_msg, get_arrow_marker_msg

DEBUG = False
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

class PathOrientationNode(Node):
    def __init__(self):
        super().__init__('path_orientation_node')
        self.detector = PathOrientationDetector(is_pcl_downsample=True, voxel_size=0.02)
        # publishers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.walls_pcl_pub = self.create_publisher(PointCloud2, '/point_cloud_walls', 10)
        self.ground_pcl_pub = self.create_publisher(PointCloud2, '/point_cloud_ground', 10)
        self.robot_heading_arrow_pub = self.create_publisher(Marker, '/robot_heading_arrow', 10)
        self.path_heading_arrow_pub = self.create_publisher(Marker, '/path_heading_arrow', 10)

        # timer
        self.create_timer(1.0, self.timer_callback)

        # variables
        self.path_deviation_angle = 0.0
        self.is_path_ending = False
        self.angular_correction_rate = 0.0
        self.avg_compute_time = 0.0

    def timer_callback(self):
        """
        execute required methods and publish as ros msgs
        """
        start_time = time.time() 
        self.path_deviation_angle = self.detector.compute_heading_angle()
        self.angular_correction_rate = self.detector.compute_angular_correction_rate(5)
        self.is_path_ending = self.detector.check_path_ending()
        self.avg_compute_time += (time.time() - start_time)
        self.avg_compute_time /= 2
        self.get_logger().info("-"*40)
        self.get_logger().info("Path deviation angle (deg): "+str(round(self.path_deviation_angle, 3)))
        self.get_logger().info("Angular correction rate (deg/s): "+str(round(self.angular_correction_rate, 3)))
        self.get_logger().info("Path ending status: "+str(self.is_path_ending))
        self.get_logger().info("Average compute time (ms): "+str(round(self.avg_compute_time*1000, 3)))

        # publish data as ros msg
        self.publish_cmd_vel()
        self.publish_pcl()
        self.publish_arrow_marker()         

    def publish_cmd_vel(self):
        """
        publish path correction angular rate(deg/s)
        through cmd_vel topic
        """
        self.get_logger().info("Publishing cmd_vel ...")
        twist_msg = Twist()
        twist_msg.linear.x = 0.1
        twist_msg.angular.z = self.angular_correction_rate 
        self.publisher_.publish(twist_msg)
    
    def publish_pcl(self):
        """
        publish walls and ground pcl data as 
        ROS2 PoinCloud2 msg type
        """
        self.get_logger().info("Publishing segmented point cloud...")
        ground_pcl_msg = generate_pointcloud2_msg(pcl_data=self.detector.ground_pcl, 
                                                pcl_fields=FIELDS_XYZ,
                                                timestamp=self.get_clock().now().to_msg(),
                                                frame_id="base_link")
        walls_pcl_msg = generate_pointcloud2_msg(pcl_data=np.vstack((self.detector.l_wall_pcl, self.detector.r_wall_pcl)),
                                                pcl_fields=FIELDS_XYZ,
                                                timestamp=self.get_clock().now().to_msg(),
                                                frame_id="base_link")
        # publishing msg
        self.ground_pcl_pub.publish(ground_pcl_msg)
        self.walls_pcl_pub.publish(walls_pcl_msg)    

    def publish_arrow_marker(self):
        """
        publish robot heading and path deviation angle as 
        ROS2 Marker msg type
        """
        self.get_logger().info("Publishing heading arrow marker...")
        robot_heading_arrow_msg = get_arrow_marker_msg(euler_heading_angle=(0, 0, 90), 
                                                    rgb_color=(1.0, 0.0, 0.0), 
                                                    timestamp=self.get_clock().now().to_msg(),
                                                    arrow_len=0.4, 
                                                    id=0,
                                                    frame_id="base_link") 
        path_heading_arrow_msg = get_arrow_marker_msg(euler_heading_angle=(0, 0, 90+self.path_deviation_angle), 
                                                    rgb_color=(0.0, 1.0, 0.0), 
                                                    timestamp=self.get_clock().now().to_msg(),
                                                    arrow_len=0.8, 
                                                    id=1,
                                                    frame_id="base_link")         
        # publishing msg
        self.robot_heading_arrow_pub.publish(robot_heading_arrow_msg)
        self.path_heading_arrow_pub.publish(path_heading_arrow_msg)         

def load_point_cloud(file_path):
        """
        load point cloud from pickle file 
        Args:
            file_path (str): path to pickle file

        Return:
            np.ndarray: point cloud from file downsampled
        """
        row_np_array = np.load(file_path)
        row_pointcloud = row_np_array['arr_0']
        # invert y axis
        row_pointcloud[:, 1] *= -1
        return row_pointcloud

def main(args=None):
    rclpy.init(args=args)
    detector_node = PathOrientationNode()
    file_path = pkg_resources.resource_filename('path_orientation_detector', '4.npz')
    pcl_data = load_point_cloud(file_path)
    detector_node.detector.set_pcl_from_array(pcl_data)

    rclpy.spin(detector_node)
    detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__": 
    main()

    


    

    


    

    













