import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'path_orientation_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # added rviz_config and launch files
        ('share/' + package_name, ['config/rviz_config.rviz']),
        ('share/' + package_name, ['launch/row_nav_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vaishanth_r',
    maintainer_email='vaishanth.rmrj@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'row_nav = path_orientation_detector.row_nav:main',
        ],
    },
)
