# neupeak-take-home-challenge
Take home challenge provided by Neupeak Robotics

## Test Description:

1. [TASK 1]
Create an algorithm that takes in the following:
    row_pointcloud : scanned pointcloud (3D numpy array) for the row from depth camera

And outputs the following:
    angular_rate : rate in deg/s the robot should turn at such that it will centre itself in the row

The goal is to genereate a target angular rate that the robot uses to turn. Positive rate means robot
will turn right, negative rate means robot will turn left. Assume the robot moves forward at some constant velocity
set by the user. Your algorithm then generates angular rates for the robot while it moves through the row. The ideal angular 
rate will make it so that the robot is always centred in the row. When the robot is perfectly centred the angular rate should be 0
since we just need the robot to go straight.

2. [TASK 2]
Implement a ROS Node that will output correction anglular rate for the robot. Publish cmd_vel to a twist
ROS topic

3. [TASK 3]
unit-test: Write a unit test script using standard Python unittest library

4. [OPTIONAL BONUS 1]
end_of_row: Boolean parameter that is True if end of row is detected

### Evaluation Criteria: Method runtime. Lower is better.

## Requirments:

- ROS2 Humble
- numpy==1.24.0
- matplotlib==3.5.1
- open3d==0.17.0
- scikit-learn==1.3.2

## Installation:
1. Install ROS2 Humble
[ROS2 Humble Install docs](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

3. Install requirements.txt
```
pip install -r requirements.txt
```

## Instructions to run
1. Without ROS
```
python row_nav.py '<input_point_cloud_file>' 
```
2. With ROS
```
cd ros2_ws/
colcon build --symlink-install
source install/setup.bash
ros2 launch path_orientation_detector row_nav_node.launch.py input_pcl_file:='<point_cloud_file_name>'
```
- Note: If the Rviz config does not load, manually load it from ros2_ws/src/path_orientation_detector/config/rviz_config.rviz

<img src="https://github.com/vaishanth-rmrj/neupeak-take-home-challenge/blob/main/git_extras/rviz_viz.png" alt="Image Alt Text" width="800" height="600" />

- Red arrow-> Robot direction
- Green arrow -> Path deviation

3. To run test script
```
python -m unittest test_path_orientation_detector.py 
```

## How does it work ?
1. Downsample point cloud for better forformance (<25ms for initial pcl loading)

<img src="https://github.com/vaishanth-rmrj/neupeak-take-home-challenge/blob/main/git_extras/open3d_pcl_viz.png" alt="Image Alt Text" width="200" height="200" />

3. Segment the point cloud to left, right walls and ground
4. Project the point cloud to 2D space thereby getting a top view
5. Apply PCA on both left and right wall 2D point to get the direction of orientation.

<img src="https://github.com/vaishanth-rmrj/neupeak-take-home-challenge/blob/main/git_extras/matplotlib_path_dev_viz.png" alt="Image Alt Text" width="200" height="50" />

7. Average the direction angles to get the path deviation angle.
8. Compute the angular correction rate using this angle.

<img src="https://github.com/vaishanth-rmrj/neupeak-take-home-challenge/blob/main/git_extras/program_output.png" alt="Image Alt Text" width="600" height="200" />
