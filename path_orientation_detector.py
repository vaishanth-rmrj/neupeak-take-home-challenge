import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import KMeans

DEBUG = False

class PathOrientationDetector:
    def __init__(self, 
                 distance_threshold=0.01, 
                 height_threshold=-0.1,
                 is_pcl_downsample=False,  
                 voxel_size=0.02) -> None:
        
        self.pcl_data = np.ndarray([])
        self.l_wall_pcl, self.r_wall_pcl = np.ndarray([]), np.ndarray([])
        self.ground_pcl = np.ndarray([])

        # params
        self.distance_threshold = distance_threshold
        self.height_threshold = height_threshold
        self.voxel_size = voxel_size
        self.is_pcl_downsample = is_pcl_downsample

        # variables
        self.is_path_ending = False
        self.path_deviation_angle = 0  

    def set_pcl_from_array(self, pcl_array):
        try:
            if not isinstance(pcl_array, np.ndarray):
                raise ValueError("Invalid point cloud type")
                
            if self.is_pcl_downsample:
                # downsample pcl for better performance
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcl_array)
                downpcd = pcd.voxel_down_sample(self.voxel_size) 
                self.pcl_data = np.asarray(downpcd.points)    
            else:
                self.pcl_data = pcl_array

            # segment pcl to walls and ground
            self.l_wall_pcl, self.r_wall_pcl, self.ground_pcl = self.pcl_segmentation_open3d(self.pcl_data, 
                                                                                            self.distance_threshold, 
                                                                                            self.height_threshold)
        except ValueError as ve:
            print(f"Error: {ve}")    

       
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

        # kmeans clustering to separate left and right walls  
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(wall_pcl)

        # separate pcl into left wall and right wall clusters
        left_wall_cluster = wall_pcl[labels == 0]
        right_wall_cluster = wall_pcl[labels == 1]

        return left_wall_cluster, right_wall_cluster, np.asarray(inlier_cloud.points)

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

    def compute_heading_angle(self, show_visualization=False):
        """
        compute heading angle using pcl data of walls
        Arguments:
            pcl_data : Point cloud data of walls
        """
        # convert 3d pcl to 2d data points by removing y-axis
        left_data_pts = self.l_wall_pcl[:, [0, 2]].copy()
        rigth_data_pts = self.r_wall_pcl[:, [0, 2]].copy()

        # compute heading direction of each wall
        # PCA to determine principal direction
        l_principal_dir = self.compute_principal_direction(left_data_pts)
        r_principal_dir = self.compute_principal_direction(rigth_data_pts)
        
        # computer heading angle using eigen values and vectors
        l_heading_angle = 90.0 - (self.get_heading_angle(l_principal_dir) * -1)
        r_heading_angle = 90.0 - (self.get_heading_angle(r_principal_dir) * -1)

        if show_visualization:
            plt.scatter(left_data_pts[:, 0], left_data_pts[:, 1])
            plt.quiver(np.mean(left_data_pts[:, 0]), np.mean(left_data_pts[:, 1]), 
                    -l_principal_dir[0], -l_principal_dir[1], 
                    angles='xy', scale_units='xy', scale=0.5, 
                    color='red', label=f'Heading Angle: {l_heading_angle:.2f} degrees')

            plt.scatter(rigth_data_pts[:, 0], rigth_data_pts[:, 1])
            plt.quiver(np.mean(rigth_data_pts[:, 0]), np.mean(rigth_data_pts[:, 1]), 
                    -r_principal_dir[0], -r_principal_dir[1], 
                    angles='xy', scale_units='xy', scale=0.1, 
                    color='green', label=f'Heading Angle: {r_heading_angle:.2f} degrees')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('PCA for Vertical Line Direction')
            plt.legend()
            plt.show()
        
        self.path_deviation_angle = (l_heading_angle + r_heading_angle)/2.0   
        return self.path_deviation_angle
    
    def compute_angular_correction_rate(self, delta_t):
        """
        computer angular correction rate (deg/s)
        Arguments:
            delta_t : time period

        Returns:
            ang_rate_deg: rate of angular deviation
        """
        # compute heading angle using pcl data of walls
        heading_angle = self.compute_heading_angle()

        # convert to rads
        heading_rads = heading_angle * (3.14159 / 180.0)
        # angular rate
        ang_rate_rad = heading_rads / delta_t
        ang_rate_deg = ang_rate_rad * (180.0 / 3.14159)
        return ang_rate_deg*-1

    def compute_bbox_area(self, bbox_pts):
        """
        compute surface area from 3D bounding box
        Arguments:
            bbox_pts (x, y, z): 3D points

        Returns:
            surface area of 3D bounding box
        """
        bbox_max_z = np.max(bbox_pts[:, 2])
        bbox_min_z = np.min(bbox_pts[:, 2])
        bbox_max_y = np.max(bbox_pts[:, 1])
        bbox_min_y = np.min(bbox_pts[:, 1])
        z_len = bbox_max_z - bbox_min_z
        y_len = bbox_max_y - bbox_min_y
        return z_len*y_len  
    
    def check_path_ending(self):
        """
        check if the path is ending my  measuring 
        left and right wall area
        Returns:
            bool: True if path ending
        """
        left_pcd = o3d.geometry.PointCloud()
        left_pcd.points = o3d.utility.Vector3dVector(self.l_wall_pcl)

        right_pcd = o3d.geometry.PointCloud()
        right_pcd.points = o3d.utility.Vector3dVector(self.r_wall_pcl)

        # # Compute the axis-aligned bounding box
        left_bbox = left_pcd.get_minimal_oriented_bounding_box()
        right_bbox = right_pcd.get_minimal_oriented_bounding_box()
        # Extract bounding box corners
        left_bbox_corners = np.asarray(left_bbox.get_box_points())
        right_bbox_corners = np.asarray(right_bbox.get_box_points())

        l_wall_area = self.compute_bbox_area(left_bbox_corners)
        r_wall_area = self.compute_bbox_area(right_bbox_corners)
        return l_wall_area < 0.1 and r_wall_area < 0.1
        
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