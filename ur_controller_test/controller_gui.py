#!/usr/bin/env python3

import os
import sys
import subprocess
import signal
import threading
import time
from enum import Enum
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from ament_index_python.packages import get_package_share_directory

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QGroupBox, QGridLayout, QMessageBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer

# Import the DataLogger class from the data_logger module
from ur_controller_test.data_logger import DataLogger
# Import our wave generator module for waveform types
from ur_controller_test.wave_generators import WaveformType


class ControllerGUI(QMainWindow):
    """GUI for configuring and running the Joint Trajectory Controller tests."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize ROS 2 node
        rclpy.init(args=None)
        self.node = Node('controller_gui')
        
        # Create an executor for ROS callbacks
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        # Initialize joint state subscriber
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_efforts = None
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Controller process
        self.controller_process = None
        self.test_running = False
        self.test_start_time = None
        
        # Data logger
        self.data_logger = None
        self.logging_enabled = False
        
        # Test parameters
        self.control_mode = "position"
        self.waveform_type = "sine"
        self.frequency = 0.2
        self.amplitude = 0.5
        self.duration = 30.0
        self.publish_rate = 10.0
        self.active_joints = [True] * 6
        self.joint_phases = [0.0] * 6  # Default phase values for each joint
        
        # Define joint names
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Set up the UI
        self.init_ui()
        
        # Timer for updating status
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        # Timer for logging data
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.log_data)
        # Will be started when logging is enabled

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('UR Controller Test GUI')
        self.setMinimumSize(800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel('UR Joint Trajectory Controller Test')
        title_label.setFont(QFont('Arial', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create content layout with parameter groups
        content_layout = QHBoxLayout()
        
        # Left column - Control parameters
        control_group = QGroupBox('Control Parameters')
        control_layout = QGridLayout(control_group)
        
        # Control mode dropdown
        control_layout.addWidget(QLabel('Control Mode:'), 0, 0)
        self.control_mode_combo = QComboBox()
        self.control_mode_combo.addItems(['position', 'velocity', 'acceleration'])
        control_layout.addWidget(self.control_mode_combo, 0, 1)
        
        # Waveform type dropdown
        control_layout.addWidget(QLabel('Waveform Type:'), 1, 0)
        self.waveform_type_combo = QComboBox()
        self.waveform_type_combo.addItems([waveform.value for waveform in WaveformType])
        control_layout.addWidget(self.waveform_type_combo, 1, 1)
        
        # Frequency input
        control_layout.addWidget(QLabel('Frequency (Hz):'), 2, 0)
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.01, 2.0)
        self.frequency_spin.setValue(0.2)
        self.frequency_spin.setSingleStep(0.05)
        control_layout.addWidget(self.frequency_spin, 2, 1)
        
        # Amplitude input
        control_layout.addWidget(QLabel('Amplitude (rad):'), 3, 0)
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.1, 1.0)
        self.amplitude_spin.setValue(0.5)
        self.amplitude_spin.setSingleStep(0.1)
        control_layout.addWidget(self.amplitude_spin, 3, 1)
        
        # Duration input
        control_layout.addWidget(QLabel('Duration (s):'), 4, 0)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(5.0, 300.0)
        self.duration_spin.setValue(30.0)
        self.duration_spin.setSingleStep(5.0)
        control_layout.addWidget(self.duration_spin, 4, 1)
        
        # Publish rate input
        control_layout.addWidget(QLabel('Publish Rate (Hz):'), 5, 0)
        self.publish_rate_spin = QDoubleSpinBox()
        self.publish_rate_spin.setRange(1.0, 100.0)
        self.publish_rate_spin.setValue(10.0)
        self.publish_rate_spin.setSingleStep(1.0)
        control_layout.addWidget(self.publish_rate_spin, 5, 1)
        
        # Waypoints per trajectory
        control_layout.addWidget(QLabel('Waypoints:'), 6, 0)
        self.waypoints_spin = QSpinBox()
        self.waypoints_spin.setRange(1, 50)
        self.waypoints_spin.setValue(5)
        control_layout.addWidget(self.waypoints_spin, 6, 1)
        
        content_layout.addWidget(control_group)
        
        # Middle column - Joint control with phases
        joint_group = QGroupBox('Joint Configuration')
        joint_layout = QGridLayout(joint_group)
        
        # Headers
        joint_layout.addWidget(QLabel('Joint'), 0, 0)
        joint_layout.addWidget(QLabel('Active'), 0, 1)
        joint_layout.addWidget(QLabel('Phase (rad)'), 0, 2)
        
        # Create joint controls and phase spinboxes
        self.joint_checkboxes = []
        self.phase_spinboxes = []
        
        for i, joint_name in enumerate(self.joint_names):
            # Joint name label
            joint_layout.addWidget(QLabel(joint_name), i+1, 0)
            
            # Active checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # All joints active by default
            joint_layout.addWidget(checkbox, i+1, 1)
            self.joint_checkboxes.append(checkbox)
            
            # Phase spinbox
            phase_spin = QDoubleSpinBox()
            phase_spin.setRange(-3.14, 3.14)  # -π to π
            phase_spin.setValue(0.0)
            phase_spin.setSingleStep(0.1)
            phase_spin.setDecimals(2)
            phase_spin.setSuffix(' rad')
            joint_layout.addWidget(phase_spin, i+1, 2)
            self.phase_spinboxes.append(phase_spin)
        
        # Preset buttons for common phase configurations
        preset_group = QGroupBox('Phase Presets')
        preset_layout = QHBoxLayout(preset_group)
        
        sync_button = QPushButton('Synchronized')
        sync_button.clicked.connect(self.set_synchronized_phases)
        preset_layout.addWidget(sync_button)
        
        alt_button = QPushButton('Alternating')
        alt_button.clicked.connect(self.set_alternating_phases)
        preset_layout.addWidget(alt_button)
        
        seq_button = QPushButton('Sequential')
        seq_button.clicked.connect(self.set_sequential_phases)
        preset_layout.addWidget(seq_button)
        
        random_button = QPushButton('Random')
        random_button.clicked.connect(self.set_random_phases)
        preset_layout.addWidget(random_button)
        
        # Add preset group to joint layout
        joint_layout.addWidget(preset_group, len(self.joint_names)+1, 0, 1, 3)
        
        content_layout.addWidget(joint_group)
        
        # Right column - Options and control buttons
        options_group = QGroupBox('Options')
        options_layout = QVBoxLayout(options_group)
        
        # Return to home checkbox
        self.return_home_check = QCheckBox('Return to Home on Shutdown')
        self.return_home_check.setChecked(True)
        options_layout.addWidget(self.return_home_check)
        
        # Data logging checkbox
        self.data_logging_check = QCheckBox('Enable Data Logging')
        self.data_logging_check.setChecked(True)
        options_layout.addWidget(self.data_logging_check)
        
        # Simulation mode checkbox
        self.simulation_mode_check = QCheckBox('Simulation Mode')
        self.simulation_mode_check.setChecked(False)
        self.simulation_mode_check.setToolTip('Enable when running in simulation to use simulation time')
        options_layout.addWidget(self.simulation_mode_check)
        
        # Status display
        status_group = QGroupBox('Status')
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel('Ready')
        status_layout.addWidget(self.status_label)
        
        # Add logo to the status section
        logo_label = QLabel()
        
        # Find the logo using ROS 2 package utilities
        try:
            package_share_dir = get_package_share_directory('ur_controller_test')
            logo_path = os.path.join(package_share_dir, 'resource', 'logo.png')
            
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    logo_label.setPixmap(scaled_pixmap)
                else:
                    raise ValueError("Logo file exists but couldn't be loaded as an image")
            else:
                raise FileNotFoundError(f"Logo not found at {logo_path}")
                
        except Exception as e:
            # Fallback to text if any error occurs
            logo_label.setText("UR\nController\nTest")
            logo_label.setFont(QFont('Arial', 12, QFont.Bold))
            
        logo_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(logo_label)
        
        options_layout.addWidget(status_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton('Apply Parameters')
        self.apply_button.clicked.connect(self.apply_parameters)
        button_layout.addWidget(self.apply_button)
        
        self.start_button = QPushButton('Start Test')
        self.start_button.clicked.connect(self.start_test)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton('Stop Test')
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        options_layout.addLayout(button_layout)
        
        # Add options to content layout
        content_layout.addWidget(options_group)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Center the window on the screen
        self.center_window()
    
    def set_synchronized_phases(self):
        """Set all phases to zero (synchronized)."""
        for spin in self.phase_spinboxes:
            spin.setValue(0.0)
    
    def set_alternating_phases(self):
        """Set alternating phases (0 and π)."""
        for i, spin in enumerate(self.phase_spinboxes):
            spin.setValue(0.0 if i % 2 == 0 else 3.14)
    
    def set_sequential_phases(self):
        """Set sequential phases (evenly distributed over 2π)."""
        num_joints = len(self.phase_spinboxes)
        for i, spin in enumerate(self.phase_spinboxes):
            # Distribute phases evenly from 0 to 2π
            phase = 2 * 3.14 * i / num_joints
            # Wrap to -π to π range
            if phase > 3.14:
                phase -= 2 * 3.14
            spin.setValue(phase)
    
    def set_random_phases(self):
        """Set random phases between -π and π."""
        import random
        for spin in self.phase_spinboxes:
            # Random value between -π and π
            phase = random.uniform(-3.14, 3.14)
            spin.setValue(phase)
    
    def center_window(self):
        """Center the window on the screen."""
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
    
    def joint_state_callback(self, msg):
        """Process joint state messages."""
        try:
            # Create dictionaries to store joint data by name
            position_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
            velocity_dict = {name: vel for name, vel in zip(msg.name, msg.velocity)}
            
            if len(msg.effort) > 0:
                effort_dict = {name: eff for name, eff in zip(msg.name, msg.effort)}
            else:
                effort_dict = {name: 0.0 for name in msg.name}
            
            # Map the joint data to our expected order
            self.joint_positions = [position_dict.get(name, 0.0) for name in self.joint_names]
            self.joint_velocities = [velocity_dict.get(name, 0.0) for name in self.joint_names]
            self.joint_efforts = [effort_dict.get(name, 0.0) for name in self.joint_names]
            
        except Exception as e:
            self.node.get_logger().error(f'Error in joint state callback: {str(e)}')
    
    def update_status(self):
        """Update the status display."""
        if self.test_running:
            if self.controller_process and self.controller_process.poll() is None:
                elapsed = time.time() - self.test_start_time
                self.status_label.setText(f'Test Running - {elapsed:.1f}s/{self.duration:.1f}s')
                
                # Check if we've exceeded the test duration (plus a small grace period)
                if elapsed > (self.duration + 5.0):
                    self.node.get_logger().warn(f'Test duration exceeded ({elapsed:.1f}s > {self.duration:.1f}s). Stopping test.')
                    self.stop_test()
            else:
                self.status_label.setText('Test Completed')
                self.test_running = False
                self.stop_button.setEnabled(False)
                self.start_button.setEnabled(True)
                if self.data_logger:
                    self.data_logger.close()
                    self.data_logger = None
                    self.log_timer.stop()
        else:
            if self.joint_positions:
                self.status_label.setText('Robot Connected - Ready')
            else:
                self.status_label.setText('Waiting for Robot Connection')
    
    def get_active_joints(self):
        """Get list of booleans indicating which joints are active."""
        return [checkbox.isChecked() for checkbox in self.joint_checkboxes]
    
    def get_joint_phases(self):
        """Get list of phase values for each joint."""
        return [spinbox.value() for spinbox in self.phase_spinboxes]
    
    def generate_waveform_values(self, t):
        """
        Generate waveform values based on the test parameters.
        
        This is used for data logging to get the commanded values.
        """
        positions = []
        velocities = []
        accelerations = []
        
        for i, active in enumerate(self.active_joints):
            if active:
                # Get phase for this joint
                joint_phase = self.joint_phases[i]
                
                # Calculate waveform values based on type with phase
                if self.waveform_type == 'sine':
                    # Sine wave with phase
                    pos = self.amplitude * math.sin(2 * math.pi * self.frequency * t + joint_phase)
                    vel = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * t + joint_phase)
                    acc = -self.amplitude * (2 * math.pi * self.frequency)**2 * math.sin(2 * math.pi * self.frequency * t + joint_phase)
                
                elif self.waveform_type == 'triangle':
                    # Triangle wave with phase
                    cycle_position = ((t * self.frequency) % 1.0 + joint_phase / (2 * math.pi)) % 1.0
                    if cycle_position < 0.5:
                        pos = self.amplitude * (4 * cycle_position - 1)
                        vel = self.amplitude * 4 * self.frequency
                    else:
                        pos = self.amplitude * (3 - 4 * cycle_position)
                        vel = -self.amplitude * 4 * self.frequency
                    acc = 0.0
                
                elif self.waveform_type == 'chirp':
                    # Chirp wave with phase
                    base_freq = 0.05
                    max_freq = self.frequency * 2
                    
                    # Calculate instantaneous frequency
                    if t < self.duration:
                        inst_freq = base_freq + (max_freq - base_freq) * (t / self.duration)
                    else:
                        inst_freq = max_freq
                    
                    # Calculate position, velocity, and acceleration for chirp wave with phase
                    phase_offset = 2 * math.pi * (base_freq * t + 0.5 * (max_freq - base_freq) * (t**2) / self.duration) + joint_phase
                    pos = self.amplitude * math.sin(phase_offset)
                    
                    # Calculate derivatives
                    inst_angular_freq = 2 * math.pi * inst_freq
                    vel = self.amplitude * inst_angular_freq * math.cos(phase_offset)
                    acc = -self.amplitude * (inst_angular_freq**2) * math.sin(phase_offset)
                
                elif self.waveform_type == 'gaussian':
                    # Gaussian pulse with phase
                    # Calculate the center of the pulse in time
                    phase_time_shift = joint_phase / (2 * math.pi * self.frequency) if self.frequency > 0 else 0
                    
                    # Determine pulse width based on frequency
                    # Lower frequency = wider pulse
                    pulse_width = 1.0 / (2.0 * self.frequency) if self.frequency > 0 else 0.5
                    
                    # For periodic behavior, modulo the time with the period
                    period = 1.0 / self.frequency if self.frequency > 0 else self.duration
                    t_periodic = t % period
                    t_center = period / 2 + phase_time_shift
                    
                    # Gaussian function centered at t_center
                    pos = self.amplitude * math.exp(-((t_periodic - t_center) ** 2) / (2 * pulse_width ** 2))
                    
                    # Calculate derivatives (velocity and acceleration)
                    # First derivative of Gaussian (velocity)
                    gaussian_factor = -2 * (t_periodic - t_center) / (2 * pulse_width ** 2)
                    vel = pos * gaussian_factor
                    
                    # Second derivative of Gaussian (acceleration)
                    acc = vel * gaussian_factor + pos * (-2 / (2 * pulse_width ** 2))
                
                elif self.waveform_type == 'sawtooth':
                    # Sawtooth wave with phase
                    # Apply phase as a cycle position offset
                    phase_fraction = joint_phase / (2 * math.pi)
                    cycle_position = ((t * self.frequency) + phase_fraction) % 1.0
                    
                    # Linear ramp from -amplitude to +amplitude
                    pos = self.amplitude * (2 * cycle_position - 1)
                    
                    # Velocity is constant throughout the cycle except at the discontinuity
                    vel = self.amplitude * 2 * self.frequency
                    
                    # Acceleration is 0 except at the discontinuity
                    acc = 0.0
                    
                    # At the discontinuity, set appropriate values for smoother motion
                    discontinuity_window = 0.01
                    if cycle_position < discontinuity_window or cycle_position > (1.0 - discontinuity_window):
                        vel = 0.0  # Avoid extreme velocity at discontinuity
                
                elif self.waveform_type == 'pulse':
                    # Pulse train with phase
                    # Apply phase as a cycle position offset
                    phase_fraction = joint_phase / (2 * math.pi)
                    cycle_position = ((t * self.frequency) + phase_fraction) % 1.0
                    
                    # Pulse width as a fraction of the period (10% duty cycle)
                    pulse_width = 0.1
                    
                    # Generate the pulse
                    if cycle_position < pulse_width:
                        pos = self.amplitude  # Pulse high
                    else:
                        pos = 0.0  # Pulse low
                    
                    # Velocity and acceleration are 0 except at transitions
                    vel = 0.0
                    acc = 0.0
                    
                    # Small non-zero values at transitions for smoother motion
                    transition_window = 0.01
                    if (abs(cycle_position) < transition_window or 
                        abs(cycle_position - pulse_width) < transition_window):
                        vel = 0.0  # Avoid extreme velocity at transitions
                
                elif self.waveform_type == 'noise':
                    # Simplified noise implementation for the GUI
                    # Since we can't maintain state between calls in this method,
                    # we use a deterministic but noisy pattern based on time and phase
                    
                    # Use a combination of sine waves at different frequencies
                    # to create a semi-random but repeatable pattern
                    noise_freq1 = self.frequency * 2.5
                    noise_freq2 = self.frequency * 3.7
                    noise_freq3 = self.frequency * 4.9
                    
                    # Generate noise using multiple sine waves with phase offset
                    pos = self.amplitude * 0.5 * (
                        math.sin(2 * math.pi * noise_freq1 * t + joint_phase) +
                        0.7 * math.sin(2 * math.pi * noise_freq2 * t + joint_phase * 1.5) +
                        0.3 * math.sin(2 * math.pi * noise_freq3 * t + joint_phase * 0.8)
                    )
                    
                    # Calculate derivatives
                    vel = self.amplitude * 0.5 * (
                        2 * math.pi * noise_freq1 * math.cos(2 * math.pi * noise_freq1 * t + joint_phase) +
                        0.7 * 2 * math.pi * noise_freq2 * math.cos(2 * math.pi * noise_freq2 * t + joint_phase * 1.5) +
                        0.3 * 2 * math.pi * noise_freq3 * math.cos(2 * math.pi * noise_freq3 * t + joint_phase * 0.8)
                    )
                    
                    acc = -self.amplitude * 0.5 * (
                        (2 * math.pi * noise_freq1)**2 * math.sin(2 * math.pi * noise_freq1 * t + joint_phase) +
                        0.7 * (2 * math.pi * noise_freq2)**2 * math.sin(2 * math.pi * noise_freq2 * t + joint_phase * 1.5) +
                        0.3 * (2 * math.pi * noise_freq3)**2 * math.sin(2 * math.pi * noise_freq3 * t + joint_phase * 0.8)
                    )
                
                else:
                    # Default to sine wave with phase
                    pos = self.amplitude * math.sin(2 * math.pi * self.frequency * t + joint_phase)
                    vel = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * t + joint_phase)
                    acc = -self.amplitude * (2 * math.pi * self.frequency)**2 * math.sin(2 * math.pi * self.frequency * t + joint_phase)
                
            else:
                # Inactive joint - use current position and zero derivatives
                pos = 0.0
                vel = 0.0
                acc = 0.0
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        return positions, velocities, accelerations
    
    def apply_parameters(self):
        """Apply the current UI parameters without starting the test."""
        # Get parameters from UI
        self.control_mode = self.control_mode_combo.currentText()
        self.waveform_type = self.waveform_type_combo.currentText()
        self.frequency = self.frequency_spin.value()
        self.amplitude = self.amplitude_spin.value()
        self.duration = self.duration_spin.value()
        self.publish_rate = self.publish_rate_spin.value()
        self.active_joints = self.get_active_joints()
        self.joint_phases = self.get_joint_phases()
        
        # Log the updated parameters
        self.node.get_logger().info(f"Parameters applied: control_mode={self.control_mode}, "
                                   f"waveform_type={self.waveform_type}, frequency={self.frequency}, "
                                   f"amplitude={self.amplitude}, duration={self.duration}, "
                                   f"publish_rate={self.publish_rate}")
        self.node.get_logger().info(f"Active joints: {self.active_joints}")
        self.node.get_logger().info(f"Joint phases: {self.joint_phases}")
        
        # Update status
        self.status_label.setText('Parameters Applied - Ready')
    
    def start_test(self):
        """Start the controller test with current parameters."""
        if self.test_running:
            return
        
        # Check if we're receiving joint states
        if not self.joint_positions:
            QMessageBox.warning(self, 'Warning', 
                               'No joint state data received. Make sure the robot is connected.')
            return
        
        # Apply parameters first
        self.apply_parameters()
        
        # Additional parameters for test execution
        waypoints = self.waypoints_spin.value()
        return_home = self.return_home_check.isChecked()
        self.logging_enabled = self.data_logging_check.isChecked()
        simulation_mode = self.simulation_mode_check.isChecked()
        
        # Format active_joints parameter properly for ROS 2
        # The parameter needs to be in a format that ROS 2 can parse as an array
        # Note that Python's str([True, False]) syntax doesn't work well with ROS parameters
        active_joints_str = "["
        for i, active in enumerate(self.active_joints):
            if i > 0:
                active_joints_str += ","
            active_joints_str += str(active).lower()  # Use lowercase true/false for ROS 2
        active_joints_str += "]"
        
        # Format joint phases parameter for ROS 2
        phase_str = "["
        for i, phase in enumerate(self.joint_phases):
            if i > 0:
                phase_str += ","
            phase_str += str(phase)  # Convert float to string
        phase_str += "]"
        
        # Create command with properly formatted parameters
        cmd = [
            'ros2', 'run', 'ur_controller_test', 'jtc_controller_test',
            '--ros-args',
            '-p', f'control_mode:={self.control_mode}',
            '-p', f'waveform_type:={self.waveform_type}',
            '-p', f'frequency:={self.frequency}',
            '-p', f'amplitude:={self.amplitude}',
            '-p', f'duration:={self.duration}',
            '-p', f'publish_rate:={self.publish_rate}',
            '-p', f'waypoints_per_trajectory:={waypoints}',
            '-p', f'active_joints:={active_joints_str}',
            '-p', f'phase:={phase_str}',  # Add phase parameter
            '-p', f'return_to_home_on_shutdown:={str(return_home).lower()}',
        ]
        
        # Add parameters for simulation mode
        if simulation_mode:
            self.node.get_logger().info("Running in simulation mode - using simulation time")
            cmd.extend([
                # Set use_sim_time parameter properly
                '-p', 'use_sim_time:=true',
                # Set some additional parameters for simulation
                '-p', 'robot_description_kinematics.use_sim_time:=true',
                '-p', 'joint_state_publisher.use_sim_time:=true',
                '-p', 'joint_trajectory_controller.use_sim_time:=true',
            ])
        
        # Start the controller process
        try:
            # Log the command for debugging
            cmd_str = ' '.join(cmd)
            self.node.get_logger().info(f"Starting controller with command: {cmd_str}")
            
            # Use a shell to help with parameter parsing
            # This can resolve some parameter passing issues
            self.controller_process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            
            # Start a thread to monitor the controller process output
            def monitor_output():
                while self.controller_process and self.controller_process.poll() is None:
                    # Read and log stdout from the controller
                    stdout_line = self.controller_process.stdout.readline()
                    if stdout_line:
                        self.node.get_logger().info(f"Controller output: {stdout_line.strip()}")
                    
                    # Read and log stderr from the controller
                    stderr_line = self.controller_process.stderr.readline()
                    if stderr_line:
                        self.node.get_logger().error(f"Controller error: {stderr_line.strip()}")
                    
                    if not stdout_line and not stderr_line:
                        break
                
                # If process exited unexpectedly, log the return code
                if self.controller_process and self.controller_process.poll() is not None:
                    self.node.get_logger().warn(f"Controller process exited with code: {self.controller_process.returncode}")
            
            # Start the monitoring thread
            monitor_thread = threading.Thread(target=monitor_output, daemon=True)
            monitor_thread.start()
            
            self.test_running = True
            self.test_start_time = time.time()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # Wait a moment to ensure the controller starts up
            time.sleep(1.0)
            
            # Check if the process is still running
            if self.controller_process.poll() is not None:
                raise Exception(f"Controller process exited prematurely with code {self.controller_process.returncode}")
            
            # Initialize data logger if logging is enabled
            if self.logging_enabled:
                self.data_logger = DataLogger(
                    self.node,
                    self.active_joints,
                    self.joint_names,
                    self.control_mode,
                    self.waveform_type,
                    self.frequency,
                    self.amplitude,
                    self.publish_rate,
                    self.joint_phases  # Add phase values to data logger
                )
                self.log_timer.start(int(1000 / self.publish_rate))  # Start logging at the same rate as publishing
            
            self.node.get_logger().info('Controller test started')
            self.status_label.setText(f'Test Starting... {"(Simulation Mode)" if simulation_mode else ""}')
            
        except Exception as e:
            self.node.get_logger().error(f'Error starting controller: {str(e)}')
            QMessageBox.critical(self, 'Error', f'Failed to start controller: {str(e)}')
            self.test_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_test(self):
        """Stop the running controller test."""
        if not self.test_running or not self.controller_process:
            return
        
        try:
            self.node.get_logger().info('Stopping controller test...')
            
            # First try to terminate gracefully with SIGINT
            try:
                # Since we're using shell=True, we need to kill the process group
                if self.controller_process.poll() is None:  # If the process is still running
                    os.killpg(os.getpgid(self.controller_process.pid), signal.SIGINT)
                    self.node.get_logger().info('Sent SIGINT to controller process')
            except Exception as e:
                self.node.get_logger().warn(f'Error sending SIGINT: {str(e)}')
            
            # Wait for graceful shutdown
            shutdown_timeout = 5  # seconds
            shutdown_start = time.time()
            
            while time.time() - shutdown_start < shutdown_timeout:
                if self.controller_process.poll() is not None:
                    self.node.get_logger().info(f'Controller process terminated with code {self.controller_process.returncode}')
                    break
                time.sleep(0.1)
            
            # If process is still running after timeout, force kill it
            if self.controller_process.poll() is None:
                self.node.get_logger().warn('Controller process did not terminate, forcing kill')
                try:
                    os.killpg(os.getpgid(self.controller_process.pid), signal.SIGKILL)
                except Exception as e:
                    self.node.get_logger().error(f'Error force killing process: {str(e)}')
            
            # Close data logger
            if self.data_logger:
                self.data_logger.close()
                self.data_logger = None
                self.log_timer.stop()
            
            self.test_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            self.node.get_logger().info('Controller test stopped')
            self.status_label.setText('Test Stopped')
            
        except Exception as e:
            self.node.get_logger().error(f'Error stopping controller: {str(e)}')
            QMessageBox.critical(self, 'Error', f'Failed to stop controller: {str(e)}')
    
    def log_data(self):
        """Log data point if logging is enabled."""
        if not self.logging_enabled or not self.data_logger or not self.test_running:
            return
        
        if not self.joint_positions or not self.joint_velocities or not self.joint_efforts:
            return
        
        # Calculate elapsed time
        if self.test_start_time:
            elapsed_time = time.time() - self.test_start_time
        else:
            elapsed_time = 0.0
        
        # Generate command values based on the current test parameters and elapsed time
        cmd_positions, cmd_velocities, cmd_accelerations = self.generate_waveform_values(elapsed_time)
        
        # Add joint offsets to positions if needed (assuming offset is at rest position)
        for i in range(len(self.joint_names)):
            if self.joint_positions[i] is not None:
                # Adjust command position to include the offset (typically the current joint position at start)
                if i < len(cmd_positions):
                    cmd_positions[i] += self.joint_positions[i] - cmd_positions[i]
        
        # Log the data
        timestamp = self.node.get_clock().now().nanoseconds / 1e9
        
        self.data_logger.log_data(
            timestamp,
            elapsed_time,
            cmd_positions,
            cmd_velocities,
            cmd_accelerations,
            self.joint_positions,
            self.joint_velocities,
            self.joint_efforts
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.test_running:
            reply = QMessageBox.question(self, 'Exit', 
                                        'A test is still running. Do you want to stop it and exit?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_test()
            else:
                event.ignore()
                return
        
        # Clean up ROS resources
        if self.node:
            self.executor.shutdown()
            self.node.destroy_node()
        rclpy.shutdown()
        
        event.accept()


def main(args=None):
    app = QApplication(sys.argv)
    gui = ControllerGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 