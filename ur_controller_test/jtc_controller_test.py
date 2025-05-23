#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import signal
import sys
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
from rclpy.parameter import Parameter
from builtin_interfaces.msg import Duration
from enum import Enum

# Import our wave generator module
from ur_controller_test.wave_generators import WaveformType, get_wave_generator


class ControlMode(Enum):
    """Enumeration of available control modes."""
    POSITION = 'position'
    VELOCITY = 'velocity'
    ACCELERATION = 'acceleration'


class JointTrajectoryControllerTest(Node):
    """
    Test node for the joint_trajectory_controller for a UR3 robot.
    
    This node allows testing different control modes and waveform patterns
    to analyze the relationship between controller input and joint state output.
    """
    
    def __init__(self):
        super().__init__('joint_trajectory_controller_test')
        
        # Joint names for UR3 robot
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Default joint limits in radians
        self.joint_limits = [
            (-2*math.pi, 2*math.pi),  # shoulder_pan_joint
            (-2*math.pi, 2*math.pi),  # shoulder_lift_joint
            (-math.pi, math.pi),      # elbow_joint - limited to ±180 degrees
            (-2*math.pi, 2*math.pi),  # wrist_1_joint
            (-2*math.pi, 2*math.pi),  # wrist_2_joint
            (-2*math.pi, 2*math.pi),  # wrist_3_joint
        ]
        
        # Set up wave generator
        self.wave_generator = None
        
        # Define home position (neutral pose)
        self.home_position = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Standard UR home position
        
        # Current joint states
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_efforts = None
        
        # For tracking test duration with wall time (more reliable with simulation)
        self.wall_start_time = None
        self.use_wall_time_for_duration = False
        
        # Declare simulation parameters
        try:
            self.declare_parameter(
                'use_sim_time',
                False,
                ParameterDescriptor(
                    description='Whether to use simulation time'
                )
            )
            self.get_logger().debug('Declared use_sim_time parameter')
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            # Parameter already declared when launched with --use-sim-time
            self.get_logger().debug('use_sim_time parameter already declared')
        
        # Declare parameters with descriptions and constraints
        self.declare_parameter(
            'control_mode', 
            'position',
            ParameterDescriptor(
                description='Control mode: position, velocity, or acceleration'
            )
        )
        
        self.declare_parameter(
            'waveform_type', 
            'sine',
            ParameterDescriptor(
                description='Test waveform pattern: sine, square, triangle, step, or chirp'
            )
        )
        
        self.declare_parameter(
            'frequency', 
            0.2,
            ParameterDescriptor(
                description='Frequency of the waveform in Hz',
                floating_point_range=[FloatingPointRange(
                    from_value=0.01, 
                    to_value=2.0
                )]
            )
        )
        
        self.declare_parameter(
            'amplitude', 
            0.5,
            ParameterDescriptor(
                description='Amplitude of the waveform in radians',
                floating_point_range=[FloatingPointRange(
                    from_value=0.1, 
                    to_value=1.0
                )]
            )
        )
        
        self.declare_parameter(
            'duration', 
            30.0,
            ParameterDescriptor(
                description='Duration of the test in seconds',
                floating_point_range=[FloatingPointRange(
                    from_value=5.0, 
                    to_value=300.0
                )]
            )
        )
        
        self.declare_parameter(
            'active_joints', 
            [True, True, True, True, True, True],
            ParameterDescriptor(
                description='List of booleans indicating which joints to test'
            )
        )
        
        self.declare_parameter(
            'offset', 
            [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            ParameterDescriptor(
                description='Offset position for each joint in radians'
            )
        )
        
        # Add phase parameter for each joint
        self.declare_parameter(
            'phase', 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ParameterDescriptor(
                description='Phase offset for each joint in radians'
            )
        )
        
        self.declare_parameter(
            'command_topic', 
            '/joint_trajectory_controller/joint_trajectory',
            ParameterDescriptor(
                description='Topic to publish trajectory commands'
            )
        )
        
        self.declare_parameter(
            'state_topic', 
            '/joint_states',
            ParameterDescriptor(
                description='Topic to subscribe for joint states'
            )
        )
        
        self.declare_parameter(
            'publish_rate', 
            10.0,
            ParameterDescriptor(
                description='Rate at which to publish trajectory commands (Hz)',
                floating_point_range=[FloatingPointRange(
                    from_value=1.0, 
                    to_value=100.0
                )]
            )
        )
        
        self.declare_parameter(
            'waypoints_per_trajectory', 
            5,
            ParameterDescriptor(
                description='Number of waypoints per trajectory'
            )
        )
        
        self.declare_parameter(
            'home_position', 
            self.home_position,
            ParameterDescriptor(
                description='Home position to return to when shutting down (radians)'
            )
        )
        
        self.declare_parameter(
            'return_to_home_on_shutdown', 
            True,
            ParameterDescriptor(
                description='Whether to return to home position on shutdown'
            )
        )
        
        # Get parameters - must get use_sim_time even if declaration failed
        try:
            self.use_sim_time = self.get_parameter('use_sim_time').value
        except Exception as e:
            self.get_logger().warning(f'Error getting use_sim_time parameter: {str(e)}. Defaulting to False.')
            self.use_sim_time = False
        
        # When in simulation mode, use wall time for duration tracking to avoid issues
        self.use_wall_time_for_duration = self.use_sim_time
        
        self.control_mode = ControlMode(self.get_parameter('control_mode').value)
        self.waveform_type = WaveformType(self.get_parameter('waveform_type').value)
        self.frequency = self.get_parameter('frequency').value
        self.amplitude = self.get_parameter('amplitude').value
        self.duration = self.get_parameter('duration').value
        self.active_joints = self.get_parameter('active_joints').value
        self.offset = self.get_parameter('offset').value
        self.phase = self.get_parameter('phase').value
        self.command_topic = self.get_parameter('command_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.waypoints_per_trajectory = self.get_parameter('waypoints_per_trajectory').value
        self.home_position = self.get_parameter('home_position').value
        self.return_to_home_on_shutdown = self.get_parameter('return_to_home_on_shutdown').value
        
        # Initialize the wave generator based on the waveform type
        self.wave_generator = get_wave_generator(self.waveform_type)
        
        # Set up parameter callback to handle dynamic parameter changes
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Create subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, 
            self.state_topic, 
            self.joint_states_callback,
            10)
        self.get_logger().info(f'Subscribed to {self.state_topic}')
        
        # Create publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            self.command_topic,
            10)
        self.get_logger().info(f'Publishing to {self.command_topic}')
        
        # Initialize test variables
        self.start_time = None
        self.elapsed_time = 0.0
        self.test_running = False
        self.shutdown_initiated = False
        
        # Create timer for publishing commands
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        self.get_logger().info('Joint Trajectory Controller Test node initialized')
        if self.use_sim_time:
            self.get_logger().info('Using simulation time')
        self.log_parameters()
    
    def log_parameters(self):
        """Log all parameters for better debugging."""
        self.get_logger().info(f'Control mode: {self.control_mode.value}')
        self.get_logger().info(f'Waveform type: {self.waveform_type.value}')
        self.get_logger().info(f'Frequency: {self.frequency} Hz')
        self.get_logger().info(f'Amplitude: {self.amplitude} rad')
        self.get_logger().info(f'Duration: {self.duration} s')
        self.get_logger().info(f'Active joints: {self.active_joints}')
        self.get_logger().info(f'Offset: {self.offset}')
        self.get_logger().info(f'Phase: {self.phase}')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        self.get_logger().info(f'Waypoints per trajectory: {self.waypoints_per_trajectory}')
        self.get_logger().info(f'Home position: {self.home_position}')
        self.get_logger().info(f'Return to home on shutdown: {self.return_to_home_on_shutdown}')
        self.get_logger().info(f'Use simulation time: {self.use_sim_time}')
    
    def parameters_callback(self, params):
        """Handle parameter changes at runtime."""
        for param in params:
            if param.name == 'use_sim_time':
                self.use_sim_time = param.value
                self.get_logger().info(f'Use simulation time updated to: {self.use_sim_time}')
            
            elif param.name == 'control_mode':
                try:
                    self.control_mode = ControlMode(param.value)
                    self.get_logger().info(f'Control mode updated to: {self.control_mode.value}')
                except ValueError:
                    self.get_logger().error(f'Invalid control mode: {param.value}')
                    return rclpy.parameter.SetParametersResult(successful=False, reason="Invalid control mode")
            
            elif param.name == 'waveform_type':
                try:
                    self.waveform_type = WaveformType(param.value)
                    # Update the wave generator when waveform type changes
                    self.wave_generator = get_wave_generator(self.waveform_type)
                    self.get_logger().info(f'Waveform type updated to: {self.waveform_type.value}')
                except ValueError:
                    self.get_logger().error(f'Invalid waveform type: {param.value}')
                    return rclpy.parameter.SetParametersResult(successful=False, reason="Invalid waveform type")
            
            elif param.name == 'frequency':
                self.frequency = param.value
                self.get_logger().info(f'Frequency updated to: {self.frequency} Hz')
            
            elif param.name == 'amplitude':
                self.amplitude = param.value
                self.get_logger().info(f'Amplitude updated to: {self.amplitude} rad')
            
            elif param.name == 'duration':
                self.duration = param.value
                self.get_logger().info(f'Duration updated to: {self.duration} s')
            
            elif param.name == 'active_joints':
                self.active_joints = param.value
                self.get_logger().info(f'Active joints updated to: {self.active_joints}')
            
            elif param.name == 'offset':
                self.offset = param.value
                self.get_logger().info(f'Offset updated to: {self.offset}')
                
            elif param.name == 'phase':
                self.phase = param.value
                self.get_logger().info(f'Phase updated to: {self.phase}')
            
            elif param.name == 'publish_rate':
                self.publish_rate = param.value
                # Update timer period
                self.timer.timer_period_ns = int(1.0 / self.publish_rate * 1e9)
                self.get_logger().info(f'Publish rate updated to: {self.publish_rate} Hz')
            
            elif param.name == 'waypoints_per_trajectory':
                self.waypoints_per_trajectory = param.value
                self.get_logger().info(f'Waypoints per trajectory updated to: {self.waypoints_per_trajectory}')
            
            elif param.name == 'home_position':
                self.home_position = param.value
                self.get_logger().info(f'Home position updated to: {self.home_position}')
            
            elif param.name == 'return_to_home_on_shutdown':
                self.return_to_home_on_shutdown = param.value
                self.get_logger().info(f'Return to home on shutdown updated to: {self.return_to_home_on_shutdown}')
        
        return rclpy.parameter.SetParametersResult(successful=True)
    
    def get_current_time(self):
        """Get current time, respecting simulation time if enabled."""
        now = self.get_clock().now()
        return now.nanoseconds / 1e9
    
    def joint_states_callback(self, msg):
        """Process joint state messages and update current joint states."""
        try:
            # Create dictionaries to store joint data by name
            position_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
            velocity_dict = {name: vel for name, vel in zip(msg.name, msg.velocity)}
            
            if len(msg.effort) > 0:
                effort_dict = {name: eff for name, eff in zip(msg.name, msg.effort)}
            else:
                effort_dict = {name: 0.0 for name in msg.name}
            
            # Map the joint data to our expected order
            self.current_joint_positions = [position_dict.get(name, 0.0) for name in self.joint_names]
            self.current_joint_velocities = [velocity_dict.get(name, 0.0) for name in self.joint_names]
            self.current_joint_efforts = [effort_dict.get(name, 0.0) for name in self.joint_names]
            
            # Start test when we receive joint states if not already started
            if not self.test_running and self.current_joint_positions and not self.shutdown_initiated:
                self.start_time = self.get_current_time()
                if self.use_wall_time_for_duration:
                    self.wall_start_time = time.time()
                    self.get_logger().info('Using wall time for duration tracking (simulation mode)')
                self.test_running = True
                self.get_logger().info('Test started')
        
        except Exception as e:
            self.get_logger().error(f'Error in joint states callback: {str(e)}')
    
    def generate_waveform(self, t):
        """Generate waveform values for all joints at time t using wave generator."""
        positions = []
        velocities = []
        accelerations = []
        
        for i in range(len(self.joint_names)):
            if self.active_joints[i]:
                # Get phase for this joint
                joint_phase = self.phase[i] if i < len(self.phase) else 0.0
                
                # Generate waveform using the wave generator
                pos, vel, acc = self.wave_generator.generate(
                    t, 
                    self.amplitude, 
                    self.frequency, 
                    joint_phase, 
                    self.duration
                )
                
                # Apply offset
                pos += self.offset[i]
            else:
                # Use current position and zero velocity/acceleration for inactive joints
                pos = self.current_joint_positions[i] if self.current_joint_positions else self.offset[i]
                vel = 0.0
                acc = 0.0
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        return positions, velocities, accelerations
    
    def generate_trajectory(self, t):
        """Generate a trajectory with the specified number of waypoints."""
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.joint_names
        
        # Time horizon for the trajectory (seconds)
        horizon = 1.0 / self.publish_rate * 2.0  # 2x the publish period
        
        # Apply a ramp-up factor during the first second to ensure smooth start
        # This helps transition from the robot's current position to the waveform
        ramp_up_duration = 1.0  # seconds
        ramp_up_factor = min(1.0, t / ramp_up_duration) if t < ramp_up_duration else 1.0
        
        # Generate waypoints
        for i in range(self.waypoints_per_trajectory):
            # Calculate time for this waypoint
            waypoint_time = t + (horizon * i / (self.waypoints_per_trajectory - 1) if self.waypoints_per_trajectory > 1 else 0)
            
            # Generate waveform values for this waypoint time
            positions, velocities, accelerations = self.generate_waveform(waypoint_time)
            
            # Apply ramp-up during the initial period to avoid tolerance violations
            if waypoint_time < ramp_up_duration:
                local_ramp_factor = min(1.0, waypoint_time / ramp_up_duration)
                
                # If we have current positions, blend from current to commanded
                if self.current_joint_positions:
                    for j in range(len(positions)):
                        # Start command = current position, Target = waveform position
                        if self.active_joints[j]:
                            waveform_pos = positions[j]
                            current_pos = self.current_joint_positions[j]
                            # Linear interpolation
                            positions[j] = current_pos + local_ramp_factor * (waveform_pos - current_pos)
                            # Scale velocities and accelerations by ramp factor
                            velocities[j] *= local_ramp_factor
                            accelerations[j] *= local_ramp_factor
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            
            # Set the appropriate control mode values
            if self.control_mode == ControlMode.POSITION:
                point.positions = positions
                if i > 0:  # Set velocities for all points except the first one for smoother motion
                    point.velocities = velocities
                if i > 1:  # Set accelerations for better trajectory following
                    point.accelerations = accelerations
            
            elif self.control_mode == ControlMode.VELOCITY:
                # In velocity mode, we still need positions to avoid discontinuities
                point.positions = positions
                point.velocities = velocities
                if i > 0:  # Set accelerations for all points except the first one
                    point.accelerations = accelerations
            
            elif self.control_mode == ControlMode.ACCELERATION:
                # In acceleration mode, we still need positions and velocities
                point.positions = positions
                point.velocities = velocities
                point.accelerations = accelerations
            
            # Calculate time from start
            time_from_start = horizon * i / (self.waypoints_per_trajectory - 1) if self.waypoints_per_trajectory > 1 else 0
            
            # Set time from start
            sec = int(time_from_start)
            nanosec = int((time_from_start - sec) * 1e9)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            
            # Add the point to the trajectory
            traj_msg.points.append(point)
        
        return traj_msg
    
    def move_to_home_position(self):
        """Send the robot to its home position."""
        if not self.current_joint_positions:
            self.get_logger().warn('Cannot move to home position: current joint positions unknown')
            return False
        
        self.get_logger().info('Moving robot to home position before shutdown...')
        
        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.joint_names
        
        # Calculate appropriate time based on the maximum joint movement
        max_movement = 0.0
        for i, (current, target) in enumerate(zip(self.current_joint_positions, self.home_position)):
            movement = abs(target - current)
            max_movement = max(max_movement, movement)
        
        # Use a conservative velocity (30 deg/s = ~0.5 rad/s) to calculate trajectory time
        safe_velocity = 0.5  # rad/s
        trajectory_time = max(2.0, max_movement / safe_velocity)  # At least 2 seconds
        
        # Create points for a smooth trajectory
        num_points = 5
        for i in range(num_points):
            alpha = i / (num_points - 1)
            
            # Linear interpolation between current position and home
            positions = []
            velocities = []
            
            for j in range(len(self.joint_names)):
                # Calculate interpolated position
                current = self.current_joint_positions[j]
                target = self.home_position[j]
                pos = current + alpha * (target - current)
                positions.append(pos)
                
                # Set velocities (zero at endpoints, non-zero in between)
                if i == 0 or i == num_points - 1:
                    vel = 0.0
                else:
                    vel = (target - current) / trajectory_time
                velocities.append(vel)
            
            # Create the trajectory point
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            
            # Set time from start
            time_from_start = trajectory_time * alpha
            sec = int(time_from_start)
            nanosec = int((time_from_start - sec) * 1e9)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            
            # Add point to trajectory
            traj_msg.points.append(point)
        
        # Publish the trajectory
        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info(f'Home trajectory sent. Duration: {trajectory_time:.2f}s')
        
        # Sleep to allow trajectory to complete (non-blocking in case of shutdown)
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # Wait for trajectory to be executed (with timeout)
        timeout = trajectory_time + 1.0  # Add 1 second buffer
        start_time = time.time()  # Use wall time for timeout
        
        while time.time() - start_time < timeout:
            # Process any pending callbacks to receive joint state updates
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if we're close enough to home position
            if self.current_joint_positions:
                max_error = 0.0
                for i, (current, target) in enumerate(zip(self.current_joint_positions, self.home_position)):
                    error = abs(target - current)
                    max_error = max(max_error, error)
                
                if max_error < 0.1:  # Within 0.1 rad (~5.7 degrees)
                    self.get_logger().info('Robot successfully reached home position')
                    return True
        
        self.get_logger().warn('Timeout waiting for robot to reach home position')
        return False
    
    def shutdown(self):
        """Perform a clean shutdown with robot returning to home."""
        if self.shutdown_initiated:
            return
        
        self.shutdown_initiated = True
        self.test_running = False
        self.get_logger().info('Shutdown initiated')
        
        # Return to home position if enabled
        if self.return_to_home_on_shutdown and self.current_joint_positions:
            try:
                self.move_to_home_position()
            except Exception as e:
                self.get_logger().error(f'Error moving to home position: {str(e)}')
        
        self.get_logger().info('Shutdown complete')
    
    def timer_callback(self):
        """Timer callback for publishing trajectory commands."""
        if self.shutdown_initiated:
            return
            
        if not self.test_running or not self.current_joint_positions:
            return
        
        current_time = self.get_current_time()
        self.elapsed_time = current_time - self.start_time
        
        # Check if test duration has elapsed using appropriate time source
        duration_exceeded = False
        
        # First check normal elapsed time (based on ROS clock which might be simulation time)
        if self.elapsed_time > self.duration:
            duration_exceeded = True
            self.get_logger().info(f'Test duration exceeded using ROS time: {self.elapsed_time:.1f} > {self.duration:.1f}s')
        
        # Also check wall time if in simulation mode for more reliable duration tracking
        if not duration_exceeded and self.use_wall_time_for_duration and self.wall_start_time is not None:
            wall_elapsed = time.time() - self.wall_start_time
            if wall_elapsed > self.duration:
                duration_exceeded = True
                self.get_logger().info(f'Test duration exceeded using wall time: {wall_elapsed:.1f} > {self.duration:.1f}s')
        
        if duration_exceeded:
            if self.test_running:
                self.get_logger().info('Test completed')
                self.test_running = False
                
                # Move to home position if the test is complete
                if self.return_to_home_on_shutdown:
                    self.move_to_home_position()
            return
        
        # Generate and publish trajectory
        trajectory = self.generate_trajectory(self.elapsed_time)
        self.trajectory_pub.publish(trajectory)
        
        # Log progress periodically
        if int(self.elapsed_time) % 5 == 0 and self.elapsed_time % 1.0 < 1.0 / self.publish_rate:
            self.get_logger().info(f'Test progress: {self.elapsed_time:.1f}/{self.duration:.1f} s')
            
            # Log first joint data for debugging
            if self.active_joints[0]:
                positions, velocities, _ = self.generate_waveform(self.elapsed_time)
                self.get_logger().info(f'Joint 0 command: pos={positions[0]:.4f}, vel={velocities[0]:.4f}')
                self.get_logger().info(f'Joint 0 state: pos={self.current_joint_positions[0]:.4f}, vel={self.current_joint_velocities[0]:.4f}')


# Signal handler for graceful shutdown
controller_test_node = None

def signal_handler(sig, frame):
    global controller_test_node
    if controller_test_node:
        print("\nReceived interrupt, shutting down gracefully...")
        controller_test_node.shutdown()
    else:
        print("\nReceived interrupt but node not initialized")


def main(args=None):
    global controller_test_node  # Use controller_test_node to match the variable used in signal_handler
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rclpy.init(args=args)
    
    try:
        controller_test_node = JointTrajectoryControllerTest()  # Assign to controller_test_node
        
        try:
            rclpy.spin(controller_test_node)
        except KeyboardInterrupt:
            pass
        finally:
            # Ensure we attempt to return to home position even on exceptions
            if controller_test_node and not controller_test_node.shutdown_initiated:
                controller_test_node.shutdown()
    
    except Exception as e:
        print(f'Exception in controller test node: {str(e)}')
    finally:
        # Clean up
        if controller_test_node:
            controller_test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
