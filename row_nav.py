import numpy as np
import matplotlib.pyplot as plt
from path_orientation_detector import PathOrientationDetector
import argparse
import time

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
        # invert y axis
        row_pointcloud[:, 1] *= -1
        return row_pointcloud


if __name__ == "__main__":
    # args parssing
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pcl_file", help="Name of the input pcl file")
    args = parser.parse_args()

    # initialize
    detector = PathOrientationDetector(is_pcl_downsample=True)    
    pcl_data = load_point_cloud(args.input_pcl_file)   
    detector.set_pcl_from_array(pcl_data)
    
    print(f"Robot heading angle: {detector.compute_heading_angle(show_visualization=True)} deg")  
    print(f"Angualr rate of deviation: {detector.compute_angular_correction_rate(5.0)} deg/s")    
    print(f"Path ending status: {detector.check_path_ending()}") 
    # Time taken: 0.026267 seconds/26.267milliseconds

    


    

    


    

    













