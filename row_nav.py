import numpy as np
import matplotlib.pyplot as plt
import pcl
import open3d as o3d

DEBUG = True

class PathOrientationDetector:
    def __init__(self, pointcloud_data_path, voxel_size=0.05) -> None:
        self.pcl_data = self.load_point_cloud(pointcloud_data_path, voxel_size)        

    def load_point_cloud(self, file_path, voxel_size):
        """
        load point cloud from pickle file and 
        downsampling using open3d

        Arguments:
            file_path: path to pickle file
            skip_factor: value to skip points

        Returns:
            row_pointcloud: point cloud from file downsampled
        """
        row_np_array = np.load(file_path)
        row_pointcloud = row_np_array['arr_0']

        # down sampling point cloud for better performance
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(row_pointcloud)
        downpcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(downpcd.points)

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
        if wall_pcl.any():
            ax.scatter(wall_pcl[:, 2], wall_pcl[:, 0], wall_pcl[:, 1], c="gray")
        if ground_pcl.any():
            ax.scatter(ground_pcl[:, 2], ground_pcl[:, 0], ground_pcl[:, 1], c="brown")
        plt.show()
    
    def pcl_segmentation(self, pcl_data, distance_threshold=0.01, height_threshold=0):        
        """
        extract wall and ground segments from pcl
        Arguments:
            pcl_data: point cloud data fetched from depth maps

        Returns:
            wall_pcl: point cloud data for wall points
            ground_pcl: point cloud data for ground points
        """
        # initialize open3d point cloud obj
        pcd = o3d.geometry.PointCloud()
        # read pcl data
        pcd.points = o3d.utility.Vector3dVector(pcl_data)

        # segment pcl to detect ground
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
        
        # plane model coefficients, for debugging
        if DEBUG:
            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # segregate ground and wall pcl points
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # visualize pcl, for debugging
        if DEBUG:
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        # convert pcl to np array
        wall_pcl = np.asarray(outlier_cloud.points)
        wall_indices = np.where(wall_pcl[:, 1] < height_threshold)
        wall_pcl = wall_pcl[wall_indices]        
        return wall_pcl, np.asarray(inlier_cloud.points)

    
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
    

    













