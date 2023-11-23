import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import KMeans

DEBUG = True

class PathOrientationDetector:
    def __init__(self, pointcloud_data_path, voxel_size=0.01) -> None:
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
        # downpcd = pcd.voxel_down_sample(voxel_size)        
        return np.asarray(pcd.points)

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
    
    def pcl_segmentation_open3d(self, pcl_data, distance_threshold=0.01, height_threshold=-0.1):        
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
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
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


    def determine_heading(self, pcl_data):
        centroid = np.mean(pcl_data, axis=0)
        cov_matrix = np.cov(pcl_data.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Use the eigenvector corresponding to the highest eigenvalue as the principal axis
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Determine heading (angle or orientation)
        heading_angle = np.arctan2(principal_axis[1], principal_axis[0])
        return np.degrees(heading_angle)

    def get_heading_angle(self, principal_direction):
        """
        get heading angle based on the pricipal
        direction of the corresponding data points

        Arguments:
            principal_direction: PCA components

        Returns:
            heading_angle for the data points
        """
        return np.degrees(np.arctan2(principal_direction[1], principal_direction[0]))
    
    def compute_principal_direction(self, data_pts):
        """
        to find the principal direction

        Arguments:
            data_pts : 2D data points from pcl
        """
        # compute covariance matrix and corresponding eigen vals
        covariance_matrix = np.cov(data_pts, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_direction = eigenvectors[:, sorted_indices[0]]
        return principal_direction

    def compute_heading_angle(self, pcl_data):
        """
        computer heading angle using pcl data of walls

        Args:
            pcl_data : Point cloud data of walls
        """
        # convert 3d pcl to 2d data points by removing y-axis
        data_pts_2d = pcl_data[:, [0, 2]].copy()

        # k-means clustering to segregate left and right wall pcl
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(data_pts_2d)

        # separate pcl into left wall and right wall clusters
        left_cluster = data_pts_2d[labels == 0]
        right_cluster = data_pts_2d[labels == 1]

        # compute heading direction of each wall
        # PCA to determine principal direction
        l_principal_dir = self.compute_principal_direction(left_cluster)
        r_principal_dir = self.compute_principal_direction(right_cluster)
        
        # computer heading angle using eigen values and vectors
        l_heading_angle = 90.0 - (self.get_heading_angle(l_principal_dir) * -1)
        r_heading_angle = 90.0 - (self.get_heading_angle(r_principal_dir) * -1)

        if DEBUG:
            plt.scatter(left_cluster[:, 0], left_cluster[:, 1])
            plt.quiver(np.mean(left_cluster[:, 0]), np.mean(left_cluster[:, 1]), 
                    -l_principal_dir[0], -l_principal_dir[1], 
                    angles='xy', scale_units='xy', scale=0.5, 
                    color='red', label=f'Heading Angle: {l_heading_angle:.2f} degrees')

            plt.scatter(right_cluster[:, 0], right_cluster[:, 1])
            plt.quiver(np.mean(right_cluster[:, 0]), np.mean(right_cluster[:, 1]), 
                    -r_principal_dir[0], -r_principal_dir[1], 
                    angles='xy', scale_units='xy', scale=0.1, 
                    color='green', label=f'Heading Angle: {r_heading_angle:.2f} degrees')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('PCA for Vertical Line Direction')
            plt.legend()
            plt.show()
        
        return (l_heading_angle + r_heading_angle)/2.0     

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

    detector = PathOrientationDetector("1.npz")
    wall_pcl, ground_pcl = detector.pcl_segmentation_open3d(detector.pcl_data)
    print(detector.compute_heading_angle(wall_pcl))

    


    

    


    

    













