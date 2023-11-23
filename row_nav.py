import numpy as np
import matplotlib.pyplot as plt
import pcl

class PathOrientationDetector:
    def __init__(self, pointcloud_data_path, skip_factor=20) -> None:
        self.pcl_data = self.load_point_cloud(pointcloud_data_path, skip_factor)        

    def load_point_cloud(self, file_path, skip_factor):
        """
        load point cloud from pickle file
        Arguments:
            file_path: path to pickle file
            skip_factor: value to skip points

        Returns:
            row_pointcloud: point cloud from file downsampled
        """
        row_np_array = np.load(file_path)
        row_pointcloud = row_np_array['arr_0']

        # down sampling point cloud for better performance
        return row_pointcloud[::skip_factor, :]

    def visualize_pcl(self, pcl_data):
        """
        visualize pcl data using matplotlib
        Arguments:
            pcl_data: point cloud data
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=0, elev=-180)
        ax.scatter(pcl_data[:, 2], pcl_data[:, 0], pcl_data[:, 1], c="green")
        plt.show()

    def visualize_pcl_segments(self, wall_pcl, ground_pcl):
        """
        visualize pcl segments using matplotlib
        Arguments:
            wall_pcl: point cloud data for wall points
            ground_pcl: point cloud data for ground points        
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=0, elev=-180)
        ax.scatter(wall_pcl[:, 2], wall_pcl[:, 0], wall_pcl[:, 1], c="gray")
        ax.scatter(ground_pcl[:, 2], ground_pcl[:, 0], ground_pcl[:, 1], c="brown")
        plt.show()
    
    def pcl_segmentation(self, pcl_data):        
        """
        extract wall and ground segments from pcl
        Arguments:
            pcl_data: point cloud data fetched from depth maps

        Returns:
            wall_pcl: point cloud data for wall points
            ground_pcl: point cloud data for ground points
        """
        # initializing pcl lib
        point_cloud = pcl.PointCloud()
        point_cloud.from_array(np.array(pcl_data, dtype=np.float32))
        
        # initializing pcl segmenter
        seg = point_cloud.make_segmenter()

        # setting model type to plane to detect ground pcl
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)

        # segmenting the pcl data 
        inliers, coefficients = seg.segment()

        # extracting ground and wall pcl data separately
        segmented_ground = point_cloud.extract(inliers, negative=False)
        segmented_walls = point_cloud.extract(inliers, negative=True)

        # converting to np array
        wall_pcl = np.asarray(segmented_walls)
        ground_pcl = np.asarray(segmented_ground)
        return wall_pcl, ground_pcl


    


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

    detector = PathOrientationDetector("4.npz")
    wall_pcl, ground_pcl = detector.pcl_segmentation(detector.pcl_data)
    detector.visualize_pcl_segments(wall_pcl, ground_pcl)

    













