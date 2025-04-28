from setuptools import find_packages, setup

package_name = 'ur_controller_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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

        ],
    },
)
