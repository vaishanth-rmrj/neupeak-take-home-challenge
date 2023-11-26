from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
import pkg_resources

package_name = 'path_orientation_detector'

def generate_launch_description():

    # getting rviz config file path
    rviz_config_file = pkg_resources.resource_filename(package_name, 'config/rviz_config.rviz')
    first_occurrence_index = rviz_config_file.find(package_name)
    second_occurrence_index = rviz_config_file.find(package_name, first_occurrence_index + 1)
    rviz_config_file = rviz_config_file[:second_occurrence_index] + rviz_config_file[second_occurrence_index + len(package_name):]
    
    return LaunchDescription([                              
        DeclareLaunchArgument('input_pcl_file', default_value='4.npz',
                              description='input point cloud file to load'),        

        Node(
            package='path_orientation_detector',
            executable='row_nav',
            name='row_nav_node',
            output='screen',
            parameters=[{'input_pcl_file': LaunchConfiguration('input_pcl_file')}],
            arguments=['--ros-args -p'],
        ),        

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        )
    ])