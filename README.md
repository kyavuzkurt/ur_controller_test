# Universal Robots Controller Test

This package contains test nodes for the Universal Robots controllers supported by ROS2 Humble.

## Installation

```bash
cd ros2_ws/src
git clone https://github.com/kyavuzkurt/ur_controller_test.git
cd ..
colcon build --packages-select ur_controller_test
```

## Usage

Launch UR Robot Drivers or UR Gazebo Drivers with joint_trajectory controller.

```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=<ur_type> initial_joint_controller:=joint_trajectory_controller
```
For simulation

```bash
ros2 launch ur_robot_driver ur_control.launch.py  ur_type:=<ur_type> initial_joint_controller:=joint_trajectory_controller robot_ip:<robot_ip>
```
For real robot

Then launch the GUI for the tester.

```bash
ros2 launch ur_controller_test controller_gui.launch.py
```


