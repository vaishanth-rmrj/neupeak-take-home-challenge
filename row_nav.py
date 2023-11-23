import numpy as np
import matplotlib.pyplot as plt
import pcl

class PathOrientationDetector:
    def __init__(self, pointcloud_data_path, skip_factor=20) -> None:
        self.pcl_data = self.load_point_cloud(pointcloud_data_path, skip_factor)        

    def load_point_cloud(self, file_path, skip_factor):
        """
        file: [string]: npz file to be processed
        return: pointcloud in numpy array
        """
        row_np_array = np.load(file_path)
        row_pointcloud = row_np_array['arr_0']

        # down sampling point cloud for better performance
        return row_pointcloud[::skip_factor, :]

    def visualize_pcl(self, pcl_data):
        """
        visualize pcl using matplotlib
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=0, elev=-180)
        ax.scatter(pcl_data[:, 2], pcl_data[:, 0], pcl_data[:, 1])
        plt.show()


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


if __name__ == "__main__":
    # pointcloud = plot_row_pointcloud("1.npz")
    orientation_detector = PathOrientationDetector("1.npz")
    orientation_detector.visualize_pcl(orientation_detector.pcl_data)