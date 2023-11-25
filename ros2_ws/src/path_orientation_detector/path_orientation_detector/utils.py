import numpy as np

# ros2 imports
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker

def generate_pointcloud2_msg(pcl_data, pcl_fields, timestamp, frame_id="base_link"):
    
    """
    generate ros2 pointcloud2 msg from pcl data
    Args:
        pcl_data (np.ndarray): point cloud data
        pcl_fields (PointField, PointField, PointField): pointfield /
            representing pcl data points 
        timestamp: timestamp for msg
        frame_id (str): base frame of marker

    Return:
        sensor_msgs.msg.PointCloud2: ros2 pointcloud2 msg
    """        
    point_cloud_data = pcl_data.copy()
    # move pcl  up by 0.1
    point_cloud_data[:, 1] += 0.1
    # swap y and z axis
    point_cloud_data[:, [1, 2]] = point_cloud_data[:, [2, 1]]
    itemsize = np.dtype(np.float32).itemsize # 32-bit float takes 4 bytes.
    # convert from ndarray to bytes
    data = point_cloud_data.astype(np.float32).tobytes()  

    # initialize PointCloud2 msg header
    header = Header()
    header.stamp = timestamp
    header.frame_id = frame_id

    # intialize PointCloud2 msg
    pcl_msg = PointCloud2()
    pcl_msg.header = header
    pcl_msg.height = 1
    pcl_msg.width = point_cloud_data.shape[0]
    pcl_msg.is_dense = False
    pcl_msg.is_bigendian = False
    pcl_msg.fields = pcl_fields
    pcl_msg.row_step = int(itemsize * 3 * point_cloud_data.shape[0])
    pcl_msg.point_step = int(itemsize * 3) # every point consists of three float32s.
    pcl_msg.data = data
    return pcl_msg

def get_quat_from_euler(yaw_degrees, pitch_degrees, roll_degrees):
    """
    convert Euler angles in degrees to quaternion.
    Args:
        yaw_degrees (float): yaw angle in degrees.
        pitch_degrees (float): pitch angle in degrees.
        roll_degrees (float): roll angle in degrees.

    Return:
        (float x, float y, float z, float w): quaternion representing the given Euler angles.
    """
    # convert degrees to radians
    yaw = np.radians(yaw_degrees)
    pitch = np.radians(pitch_degrees)
    roll = np.radians(roll_degrees)

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return qz, qy, qx, qw

def get_arrow_marker_msg(euler_heading_angle, rgb_color, timestamp, arrow_len=0.4, id=0, frame_id="base_link"):
    """
    create a ros arrow marker based
    Args:
        euler_heading_angle (int, int, int): yaw, pitch, roll in degrees
        rgb_color (float, float, float): arrow color
        timestamp: timestamp for msg
        arrow_len (float): length of the arrow marker
        id (int): marker ID
        frame_id (str): base frame of marker

    Return:
        visualization_msgs.msg.Marker: arrow marker ros msg
    """
    marker = Marker()
    marker.header.frame_id = frame_id  
    marker.header.stamp = timestamp
    marker.id = id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Set the pose of the arrow
    marker.pose.position = Point()  # Set the position of the arrow
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.25

    marker.pose.orientation = Quaternion()  # Set the orientation of the arrow
    x, y, z = euler_heading_angle
    qx, qy, qz, qw  = get_quat_from_euler(x, y, z)
    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw

    # Set the scale of the arrow
    marker.scale = Vector3()  # Set the scale of the arrow (length, diameter, diameter)
    marker.scale.x = arrow_len
    marker.scale.y = 0.03
    marker.scale.z = 0.03

    # Set the color of the arrow (RGBA)
    marker.color.r = rgb_color[0]
    marker.color.g = rgb_color[1]
    marker.color.b = rgb_color[2]
    marker.color.a = 1.0

    # Set the lifetime of the arrow marker
    marker.lifetime.sec = 1
    return marker