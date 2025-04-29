#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import os

def generate_launch_description():
    """
    Generate launch description for the controller GUI.
    
    This launch file starts the GUI application for the UR controller test.
    """
    # Define arguments
    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value='logs',
        description='Directory to store log files'
    )
    
    # Create logs directory
    logs_setup = ExecuteProcess(
        cmd=['mkdir', '-p', LaunchConfiguration('log_dir')],
        name='logs_setup',
        output='screen'
    )
    
    # Launch the GUI
    gui_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'ur_controller_test', 'controller_gui'],
        name='controller_gui',
        output='screen'
    )
    
    # Define the launch description
    ld = LaunchDescription()
    
    # Add the log directory argument
    ld.add_action(log_dir_arg)
    
    # Create logs directory
    ld.add_action(logs_setup)
    
    # Add a helpful message
    ld.add_action(LogInfo(
        msg=["Launching UR Controller Test GUI"]
    ))
    
    # Add the GUI process
    ld.add_action(gui_cmd)
    
    return ld 