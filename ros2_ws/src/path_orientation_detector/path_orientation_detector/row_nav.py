import numpy as np
import matplotlib.pyplot as plt
from .orientation_detector import PathOrientationDetector

# ros2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import pkg_resources

DEBUG = False

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

class PathOrientationNode(Node):
    def __init__(self):
        super().__init__('path_orientation_node')
        self.detector = PathOrientationDetector()
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.walls_pcl_pub = self.create_publisher(PointCloud2, '/point_cloud_walls', 10)
        self.ground_pcl_pub = self.create_publisher(PointCloud2, '/point_cloud_ground', 10)

        self.timer = self.create_timer(1.0, self.publish_cmd_vel)
        self.pcl_pub_timer = self.create_timer(1.0, self.publish_pcl)


    def publish_cmd_vel(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.2  # Change this value as needed
        twist_msg.angular.z = 0.1  # Change this value as needed
        self.publisher_.publish(twist_msg)
        self.get_logger().info("Publishing to /cmd_vel")
        self.get_logger().info(str(self.detector.compute_heading_angle()))
        self.get_logger().info(str(self.detector.compute_angular_correction_rate(5)))
        self.get_logger().info(str(self.detector.check_path_ending()))
    
    def publish_pcl(self):
        """
        publish walls and ground pcl data as 
        ROS2 PoinCloud2 msg type
        """
        ground_pcl_msg = self.generate_pointcloud2_msg(self.detector.ground_pcl)
        walls_pcl_msg = self.generate_pointcloud2_msg(np.vstack((self.detector.l_wall_pcl, self.detector.r_wall_pcl)))

        self.ground_pcl_pub.publish(ground_pcl_msg)
        self.walls_pcl_pub.publish(walls_pcl_msg)

    
    def generate_pointcloud2_msg(self, pcl_data):
        """
        generate ros2 pointcloud2 msg from pcl data
        Args:
            pcl_data: point cloud data

        Returns:
            pcl_msg: ros2 pointcloud2 msg
        """        
        point_cloud_data = pcl_data.copy()
        # invert y axis and move up by 0.1
        point_cloud_data[:, 1] *= -1
        point_cloud_data[:, 1] += 0.1
        # swap y and z axis
        point_cloud_data[:, [1, 2]] = point_cloud_data[:, [2, 1]]
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize # 32-bit float takes 4 bytes.
        # convert from ndarray to bytes
        data = point_cloud_data.astype(dtype).tobytes()  

        # initialize PointCloud2 msg header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        # intialize PointCloud2 msg
        pcl_msg = PointCloud2()
        pcl_msg.header = header
        pcl_msg.height = 1
        pcl_msg.width = point_cloud_data.shape[0]
        pcl_msg.is_dense = False
        pcl_msg.is_bigendian = False
        pcl_msg.fields = FIELDS_XYZ
        pcl_msg.row_step = int(itemsize * 3 * point_cloud_data.shape[0])
        pcl_msg.point_step = int(itemsize * 3) # every point consists of three float32s.
        pcl_msg.data = data
        return pcl_msg
         

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

    


    

    


    

    













