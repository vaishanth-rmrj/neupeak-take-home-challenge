import unittest
import numpy as np
from path_orientation_detector import PathOrientationDetector 

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

class TestPathOrientationDetector(unittest.TestCase):

    def setUp(self):
        # initialize PathOrientationDetector with default parameters
        self.detector = PathOrientationDetector(show_viz=False)
        
    
    def test_set_pcl_from_array(self):
        pcl_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.detector.set_pcl_from_array(pcl_data)
        np.testing.assert_array_equal(self.detector.pcl_data, pcl_data)

        # testing type error
        with self.assertRaises(TypeError):
           self.detector.set_pcl_from_array(10) # int input
           self.detector.set_pcl_from_array(-1) # neg num input
           self.detector.set_pcl_from_array("array") # string input

        # testing value error
        with self.assertRaises(ValueError):
            self.detector.set_pcl_from_array(np.array([])) # empty ndarray
            self.detector.set_pcl_from_array(np.array([1, 2, 3, 4]))# 1d array
            self.detector.set_pcl_from_array(np.array([[[1, 2, 3]], [[1, 2, 3]]]))# 3d array
            

    def test_pcl_segmentation_open3d(self):
        # test if pcl_segmentation_open3d returns the expected wall and ground point clouds
        pcl_data = load_point_cloud('2.npz')
        d_thres, h_thres = 0.01, -0.1 
        l_wall, r_wall, ground = self.detector.pcl_segmentation_open3d(pcl_data, d_thres, h_thres)

        # check if all point clouds are generated
        self.assertIsNotNone(l_wall)
        self.assertIsNotNone(r_wall)
        self.assertIsNotNone(ground)     

        # testing type error
        with self.assertRaises(TypeError):
           self.detector.pcl_segmentation_open3d(10, d_thres, h_thres) # int input
           self.detector.pcl_segmentation_open3d(-1, d_thres, h_thres) # neg num input
           self.detector.pcl_segmentation_open3d("array", d_thres, h_thres) # string input
           self.detector.pcl_segmentation_open3d([1, 2, 3], d_thres, h_thres) # list input

        # testing value error
        with self.assertRaises(ValueError):
            self.detector.pcl_segmentation_open3d(np.array([]), d_thres, h_thres) # empty ndarray
            self.detector.pcl_segmentation_open3d(np.array([1, 2, 3, 4]), d_thres, h_thres)# 1d array
            self.detector.pcl_segmentation_open3d(np.array([[[1, 2, 3]], [[1, 2, 3]]]), d_thres, h_thres)# 3d array
    
    def test_get_heading_angle_from_pca_dir(self):
        # test if get_heading_angle_from_pca_dir returns the heading angle in 0-360 deg range
        principal_direction = np.array([1, 1])  # assuming pca values
        heading_angle = self.detector.get_heading_angle_from_pca_dir(principal_direction)

        # check the result
        self.assertIsInstance(heading_angle, float)
        self.assertGreaterEqual(heading_angle, 0.0)
        self.assertLessEqual(heading_angle, 360.0)
    
    def test_compute_principal_direction(self):
        # test if compute_principal_direction returns the correct principal direction
        data_pts = np.array([[1, 2], [3, 4], [5, 6]])  # assuming 2D data points from pcl
        principal_direction = self.detector.compute_principal_direction(data_pts)

        # chech for the type and dim
        self.assertIsInstance(principal_direction, np.ndarray)
        self.assertEqual(principal_direction.shape, (2,))

        # testing type error
        with self.assertRaises(TypeError):
           self.detector.compute_principal_direction(10) # int input
           self.detector.compute_principal_direction(-1) # neg num input
           self.detector.compute_principal_direction("array") # string input
           self.detector.compute_principal_direction([1, 2, 3]) # list input

        # testing value error
        with self.assertRaises(ValueError):
            self.detector.compute_principal_direction(np.array([])) # empty ndarray
            self.detector.compute_principal_direction(np.array([1, 2, 3, 4]))# 1d array
            self.detector.compute_principal_direction(np.array([[[1, 2, 3]], [[1, 2, 3]]]))# 3d array
    
    def test_compute_heading_angle(self):
        # test if compute_heading_angle returns a valid heading angle
        pcl_data = load_point_cloud('3.npz')
        self.detector.set_pcl_from_array(pcl_data)
        heading_angle = self.detector.compute_heading_angle()

        # perform assertions based on result
        self.assertIsInstance(heading_angle, float)

        # testing value error
        # empty pcl arrays
        self.detector.l_wall_pcl, self.detector.r_wall_pcl = np.array([]), np.array([])
        with self.assertRaises(ValueError):
            heading_angle = self.detector.compute_heading_angle()

        # 1d arrays as input
        self.detector.l_wall_pcl, self.detector.r_wall_pcl = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
        with self.assertRaises(ValueError):
            heading_angle = self.detector.compute_heading_angle()
        
        # 2d arrays as input
        self.detector.l_wall_pcl, self.detector.r_wall_pcl = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
        with self.assertRaises(ValueError):
            heading_angle = self.detector.compute_heading_angle()

    def test_compute_angular_correction_rate(self):
        pcl_data = load_point_cloud('3.npz')  
        self.detector.set_pcl_from_array(pcl_data)
        delta_t = 5.0
        angular_correction_rate = self.detector.compute_angular_correction_rate(delta_t)

        # check if result is float
        self.assertIsInstance(angular_correction_rate, float)

        # testing zero div error
        with self.assertRaises(ZeroDivisionError):
           self.detector.compute_angular_correction_rate(0) # int input

        # testing type error
        with self.assertRaises(TypeError):
           self.detector.compute_angular_correction_rate(1) # int input
        
        # testing type error
        with self.assertRaises(ValueError):
           self.detector.compute_angular_correction_rate(-5.0) # int input
    
    def test_compute_bbox_area(self):

        # testing type error
        with self.assertRaises(TypeError):
           self.detector.compute_bbox_area(10) # int input
           self.detector.compute_bbox_area(-1) # neg num input
           self.detector.compute_bbox_area("array") # string input
           self.detector.compute_bbox_area([1, 2, 3]) # list input

        # testing value error
        with self.assertRaises(ValueError):
            self.detector.compute_bbox_area(np.array([])) # empty ndarray
            self.detector.compute_bbox_area(np.array([1, 2, 3, 4]))# 1d array
            self.detector.compute_bbox_area(np.array([[[1, 2]], [[1, 2]]]))# 2d array
    
    def test_check_path_ending(self):
        # test if check_path_ending correctly determines if the path is ending
        # Assuming minimal bounding box areas for left and right walls
        pcl_data = load_point_cloud('2.npz')  
        self.detector.set_pcl_from_array(pcl_data)
        self.assertTrue(self.detector.check_path_ending())

        # Assuming larger bounding box areas for left and right walls
        pcl_data = load_point_cloud('3.npz')  
        self.detector.set_pcl_from_array(pcl_data)
        self.assertFalse(self.detector.check_path_ending())

        # empty pcl arrays
        self.detector.l_wall_pcl, self.detector.r_wall_pcl = np.array([]), np.array([])
        with self.assertRaises(ValueError):
            self.detector.check_path_ending()

        # 1d arrays as input
        self.detector.l_wall_pcl, self.detector.r_wall_pcl = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
        with self.assertRaises(ValueError):
            self.detector.check_path_ending()
        
        # 2d arrays as input
        self.detector.l_wall_pcl, self.detector.r_wall_pcl = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
        with self.assertRaises(ValueError):
            self.detector.check_path_ending()
        
    def test_visualize_pcl(self):
        # test if visualize_pcl throws error on invalid input type
        # testing type error
        with self.assertRaises(TypeError):
           self.detector.visualize_pcl(10) # int input
           self.detector.visualize_pcl(-1) # neg num input
           self.detector.visualize_pcl("array") # string input
           self.detector.visualize_pcl([1, 2, 3]) # list input

        # testing value error
        with self.assertRaises(ValueError):
            self.detector.visualize_pcl(np.array([])) # empty ndarray
            self.detector.visualize_pcl(np.array([1, 2, 3, 4]))# 1d array
            self.detector.visualize_pcl(np.array([[[1, 2]], [[1, 2]]]))# 2d array

    def test_visualize_pcl_segments(self):
        # Test if visualize_pcl_segments throws error on invalid input type
        with self.assertRaises(TypeError):
           self.detector.visualize_pcl_segments(10, 10) # int input
           self.detector.visualize_pcl_segments(-1, -2) # neg num input
           self.detector.visualize_pcl_segments("array", "not_array") # string input
           self.detector.visualize_pcl_segments([1, 2, 3], [1, 2, 3]) # list input

        # testing value error
        with self.assertRaises(ValueError):
            self.detector.visualize_pcl_segments(np.array([]), np.array([])) # empty ndarray
            self.detector.visualize_pcl_segments(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))# 1d array
            self.detector.visualize_pcl_segments(np.array([[[1, 2]], [[1, 2]]]), np.array([[[1, 2]], [[1, 2]]]))# 2d array   


if __name__ == "__main__":
     unittest.main()
