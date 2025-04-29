from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur_controller_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.launch.py'))),
        # Include logo and other resources
        (os.path.join('share', package_name, 'resource'),
         glob(os.path.join('resource', '*.png'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kadir Yavuz Kurt',
    maintainer_email='k.yavuzkurt1@gmail.com',
    description='Testing functions for understanding limits of the Universal Robots supported ROS2 controllers.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jtc_controller_test = ur_controller_test.jtc_controller_test:main',
            'controller_gui = ur_controller_test.controller_gui:main',
        ],
    },
)
