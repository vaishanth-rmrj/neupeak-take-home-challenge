import numpy as np
import matplotlib.pyplot as plt
from .orientation_detector import PathOrientationDetector

# ros2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pkg_resources

DEBUG = False

class PathOrientationNode(Node):
    def __init__(self):
        super().__init__('path_orientation_node')
        self.detector = PathOrientationDetector()
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(1.0, self.publish_cmd_vel)

    def publish_cmd_vel(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.2  # Change this value as needed
        twist_msg.angular.z = 0.1  # Change this value as needed
        self.publisher_.publish(twist_msg)
        self.get_logger().info("Publishing to /cmd_vel")
        self.get_logger().info(str(self.detector.compute_heading_angle()))
        self.get_logger().info(str(self.detector.compute_angular_correction_rate(5)))
        self.get_logger().info(str(self.detector.check_path_ending()))

def plot_row_pointcloud(file):    
    """
    file: [string]: npz file to be processed
    return: pointcloud in numpy array
    """
    row_np_array = np.load(file)
    row_pointcloud = row_np_array['arr_0']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=0, elev=-180)
    ax.scatter(row_pointcloud[:, 2], row_pointcloud[:, 0], row_pointcloud[:, 1])
    plt.show()

    return row_pointcloud

def load_point_cloud(file_path):
        """
        load point cloud from pickle file 
        Arguments:
            file_path: path to pickle file

        Returns:
            row_pointcloud: point cloud from file downsampled
        """
        row_np_array = np.load(file_path)
        row_pointcloud = row_np_array['arr_0']
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

    


    

    


    

    













