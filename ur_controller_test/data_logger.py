#!/usr/bin/env python3

import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node


class DataLogger:
    """
    Utility for logging controller test data to CSV files.
    
    This class handles logging of joint trajectory controller inputs and joint state outputs
    to CSV files for later analysis. The logged data includes controller commands and 
    actual joint states (positions, velocities, efforts).
    """
    
    def __init__(self, node, active_joints, joint_names, control_mode, waveform_type, 
                 frequency, amplitude, publish_rate, joint_phases=None):
        """
        Initialize the data logger.
        
        Args:
            node (Node): The ROS 2 node instance for logging
            active_joints (list): List of booleans indicating which joints are active
            joint_names (list): List of joint names
            control_mode (str): Control mode being used (position, velocity, acceleration)
            waveform_type (str): Type of waveform being tested
            frequency (float): Frequency of the waveform in Hz
            amplitude (float): Amplitude of the waveform
            publish_rate (float): Rate at which commands are published
            joint_phases (list): Optional list of phase values for each joint in radians
        """
        self.node = node
        self.active_joints = active_joints
        self.joint_names = joint_names
        self.control_mode = control_mode
        self.waveform_type = waveform_type
        self.frequency = frequency
        self.amplitude = amplitude
        self.publish_rate = publish_rate
        self.joint_phases = joint_phases if joint_phases is not None else [0.0] * len(joint_names)
        
        # File handling
        self.file = None
        self.writer = None
        self.log_path = None
        self.headers = self._create_headers()
        
        # Initialize the log file
        self._init_log_file()
    
    def _create_headers(self):
        """Create CSV headers based on active joints and all joint states."""
        headers = ['timestamp', 'elapsed_time']
        
        # Add command headers for active joints
        for i, joint_name in enumerate(self.joint_names):
            if self.active_joints[i]:
                headers.append(f'{joint_name}_cmd_pos')
                headers.append(f'{joint_name}_cmd_vel')
                headers.append(f'{joint_name}_cmd_acc')
                headers.append(f'{joint_name}_phase')  # Add phase to headers
        
        # Add state headers for all joints
        for joint_name in self.joint_names:
            headers.append(f'{joint_name}_pos')
            headers.append(f'{joint_name}_vel')
            headers.append(f'{joint_name}_eff')
        
        return headers
    
    def _init_log_file(self):
        """Initialize the log file with appropriate naming and headers."""
        # Create logs directory if it doesn't exist
        # Try a few different locations to ensure we can write logs
        possible_log_dirs = [
            # Try workspace root logs dir first
            os.path.join(os.getcwd(), 'logs'),
            # Then package directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs'),
            # Finally, user's home directory
            os.path.join(os.path.expanduser('~'), 'ur_controller_logs')
        ]
        
        # Use the first directory that exists or can be created
        for log_dir in possible_log_dirs:
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Test if we can write to this directory
                test_file = os.path.join(log_dir, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                break
            except (IOError, PermissionError):
                self.node.get_logger().warning(f"Cannot write to log directory: {log_dir}")
                if log_dir == possible_log_dirs[-1]:
                    self.node.get_logger().error("Could not find a writable log directory!")
                    return
        
        # Create filename based on test parameters
        # Add a summary of phases if they're not all zero
        phase_str = ""
        if any(abs(phase) > 0.001 for phase in self.joint_phases):
            # If at least one phase is non-zero, add a phase signature to the filename
            # Use the variance of the phases as a simple signature
            phase_variance = sum((p - sum(self.joint_phases)/len(self.joint_phases))**2 for p in self.joint_phases) / len(self.joint_phases)
            phase_str = f"-phase_var_{phase_variance:.2f}"
        
        filename = f"{self.control_mode}-{self.waveform_type}-freq_{self.frequency:.2f}-amp_{self.amplitude:.2f}-pubrate_{self.publish_rate:.1f}{phase_str}.csv"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        self.log_path = os.path.join(log_dir, filename)
        
        # Open file and write headers
        try:
            self.file = open(self.log_path, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.headers)
            
            self.node.get_logger().info(f"Data logging started: {self.log_path}")
            
            # Log phase information for reference
            self.node.get_logger().info(f"Joint phases: {self.joint_phases}")
            
        except IOError as e:
            self.node.get_logger().error(f"Failed to open log file: {str(e)}")
            self.file = None
            self.writer = None
    
    def log_data(self, timestamp, elapsed_time, cmd_positions, cmd_velocities, cmd_accelerations, 
                 joint_positions, joint_velocities, joint_efforts):
        """
        Log a single data point to the CSV file.
        
        Args:
            timestamp (float): ROS time when the data was recorded
            elapsed_time (float): Time elapsed since the start of the test
            cmd_positions (list): Commanded positions for each joint
            cmd_velocities (list): Commanded velocities for each joint
            cmd_accelerations (list): Commanded accelerations for each joint
            joint_positions (list): Actual joint positions
            joint_velocities (list): Actual joint velocities
            joint_efforts (list): Actual joint efforts
        """
        if self.file is None or self.writer is None:
            self.node.get_logger().error("Cannot log data: log file not initialized")
            return
        
        # Start with timestamp and elapsed time
        row = [timestamp, elapsed_time]
        
        # Add commands for active joints
        for i, active in enumerate(self.active_joints):
            if active:
                row.append(cmd_positions[i])
                row.append(cmd_velocities[i])
                row.append(cmd_accelerations[i])
                row.append(self.joint_phases[i])  # Add phase to logged data
        
        # Add state data for all joints
        for i in range(len(self.joint_names)):
            row.append(joint_positions[i])
            row.append(joint_velocities[i])
            row.append(joint_efforts[i])
        
        # Write the row to the CSV file
        self.writer.writerow(row)
        
        # Ensure data is written to disk by flushing occasionally
        # This is a trade-off between performance and data safety
        if elapsed_time % 5.0 < 0.1:  # Flush approximately every 5 seconds
            self.file.flush()
    
    def close(self):
        """Close the log file properly."""
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None
            self.node.get_logger().info(f"Data logging completed: {self.log_path}") 