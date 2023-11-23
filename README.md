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

## Installation:

1. Python PCL libs (for ubuntu 22.04):
```
sudo apt-get update
sudo apt-get -y install python3-pcl
```
