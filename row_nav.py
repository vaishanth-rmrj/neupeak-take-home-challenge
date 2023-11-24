import numpy as np
import matplotlib.pyplot as plt
from path_orientation_detector import PathOrientationDetector

DEBUG = False

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

if __name__ == "__main__":
    detector = PathOrientationDetector()
    pcl_data = load_point_cloud("2.npz")
    detector.set_pcl_from_array(pcl_data)
    print("Robot heading angle: ", detector.compute_heading_angle())  
    print("Angualr rate of deviation: ", detector.compute_angular_correction_rate(5))    
    print("Is path ending: ", detector.check_path_ending()) 

    


    

    


    

    













