#!/usr/bin/env python3

import math
import random
from abc import ABC, abstractmethod
from enum import Enum


class WaveformType(Enum):
    """Enumeration of available test waveform patterns."""
    SINE = 'sine'
    TRIANGLE = 'triangle'
    CHIRP = 'chirp'
    GAUSSIAN = 'gaussian'
    SAWTOOTH = 'sawtooth'
    PULSE = 'pulse'
    NOISE = 'noise'


class WaveGenerator(ABC):
    """
    Abstract base class for waveform generators.
    
    This defines the interface that all concrete wave generator implementations must follow.
    """
    
    @abstractmethod
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """
        Generate a single point on the waveform at time t.
        
        Args:
            t (float): The time value
            amplitude (float): The amplitude of the waveform
            frequency (float): The frequency of the waveform in Hz
            phase (float): The phase offset in radians
            duration (float, optional): The total duration of the waveform (needed for some types)
            
        Returns:
            tuple: (position, velocity, acceleration)
        """
        pass


class SineWaveGenerator(WaveGenerator):
    """Generates a sine wave with phase control."""
    
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a sine wave point at time t."""
        # Apply phase directly to the sine function argument
        # This ensures proper offset while maintaining continuity
        angular_freq = 2 * math.pi * frequency
        
        # Start at 0 amplitude at t=0 (sin(phase) instead of sin(0+phase))
        # This ensures we don't get a sudden jump at the start
        pos = amplitude * math.sin(angular_freq * t + phase)
        
        # Velocity and acceleration derivatives respect the phase
        vel = amplitude * angular_freq * math.cos(angular_freq * t + phase)
        acc = -amplitude * (angular_freq**2) * math.sin(angular_freq * t + phase)
        
        return pos, vel, acc


class TriangleWaveGenerator(WaveGenerator):
    """Generates a triangle wave with phase control."""
    
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a triangle wave point at time t."""
        # Apply phase directly to the cycle position
        phase_fraction = phase / (2 * math.pi)
        cycle_position = ((t * frequency) + phase_fraction) % 1.0
        
        # Calculate position, velocity based on cycle position
        if cycle_position < 0.5:
            # Rising edge: 0 to 0.5 -> -amplitude to +amplitude
            pos = amplitude * (4 * cycle_position - 1)
            vel = amplitude * 4 * frequency
        else:
            # Falling edge: 0.5 to 1.0 -> +amplitude to -amplitude
            pos = amplitude * (3 - 4 * cycle_position)
            vel = -amplitude * 4 * frequency
        
        # Acceleration is 0 except at turning points
        acc = 0.0
        
        return pos, vel, acc


class ChirpWaveGenerator(WaveGenerator):
    """Generates a chirp wave (increasing frequency sine) with phase control."""
    
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a chirp wave point at time t."""
        if duration is None:
            # Default duration if not provided
            duration = 30.0
        
        # Define frequency limits
        base_freq = 0.05
        max_freq = frequency * 2
        
        # Calculate instantaneous frequency
        if t < duration:
            inst_freq = base_freq + (max_freq - base_freq) * (t / duration)
        else:
            inst_freq = max_freq
        
        # Calculate the phase term with applied phase offset
        phase_term = 2 * math.pi * (base_freq * t + 
                                   0.5 * (max_freq - base_freq) * (t**2) / duration) + phase
        
        # Calculate position for chirp wave with phase
        pos = amplitude * math.sin(phase_term)
        
        # Calculate derivatives
        inst_angular_freq = 2 * math.pi * inst_freq
        vel = amplitude * inst_angular_freq * math.cos(phase_term)
        acc = -amplitude * (inst_angular_freq**2) * math.sin(phase_term)
        
        return pos, vel, acc


class GaussianWaveGenerator(WaveGenerator):
    """Generates a Gaussian pulse wave with phase control."""
    
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a Gaussian pulse at time t."""
        if duration is None:
            # Default duration
            duration = 30.0
        
        # Phase shifts the center of the pulse as a fraction of the period
        period = 1.0 / frequency if frequency > 0 else duration
        phase_shift = phase / (2 * math.pi) * period
        
        # Determine pulse width based on frequency
        # Lower frequency = wider pulse
        pulse_width = 1.0 / (2.0 * frequency) if frequency > 0 else 0.5
        
        # For periodic behavior, modulo the time with the period
        t_periodic = t % period
        t_center = period / 2 + phase_shift  # Center shifted by phase
        
        # Wrap t_center to ensure it's within the period
        t_center = t_center % period
        
        # Handle edge cases for periodic wrapping
        delta = t_periodic - t_center
        if abs(delta) > period/2:  # If the distance is more than half a period, it's shorter to go the other way
            if delta > 0:
                delta -= period
            else:
                delta += period
        
        # Gaussian function centered at t_center with phase-adjusted position
        gaussian_factor = -(delta ** 2) / (2 * pulse_width ** 2)
        pos = amplitude * math.exp(gaussian_factor)
        
        # Calculate derivatives (velocity and acceleration)
        # First derivative of Gaussian (velocity)
        vel_factor = -delta / (pulse_width ** 2)
        vel = pos * vel_factor
        
        # Second derivative of Gaussian (acceleration)
        acc = vel * vel_factor + pos * (-1 / (pulse_width ** 2))
        
        return pos, vel, acc


class SawtoothWaveGenerator(WaveGenerator):
    """Generates a sawtooth wave with phase control."""
    
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a sawtooth wave point at time t."""
        # Apply phase directly as a cycle position offset
        phase_fraction = phase / (2 * math.pi)
        cycle_position = ((t * frequency) + phase_fraction) % 1.0
        
        # Linear ramp from -amplitude to +amplitude
        pos = amplitude * (2 * cycle_position - 1)
        
        # Velocity is constant throughout the cycle except at the discontinuity
        vel = amplitude * 2 * frequency
        
        # Acceleration is 0 except at the discontinuity
        acc = 0.0
        
        # At the discontinuity, set appropriate values
        # This improves trajectory following by avoiding extreme values
        discontinuity_window = 0.01
        if cycle_position < discontinuity_window or cycle_position > (1.0 - discontinuity_window):
            vel = 0.0  # Avoid extreme velocity at discontinuity
        
        return pos, vel, acc


class PulseWaveGenerator(WaveGenerator):
    """Generates a pulse train with phase control."""
    
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a pulse train point at time t."""
        # Apply phase directly as a cycle position offset
        phase_fraction = phase / (2 * math.pi)
        cycle_position = ((t * frequency) + phase_fraction) % 1.0
        
        # Pulse width as a fraction of the period (10% duty cycle)
        pulse_width = 0.1
        
        # Generate the pulse
        if cycle_position < pulse_width:
            pos = amplitude  # Pulse high
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
        
        return pos, vel, acc


class NoiseWaveGenerator(WaveGenerator):
    """Generates a random noise signal with frequency-based filtering."""
    
    def __init__(self):
        super().__init__()
        self.last_pos = 0.0
        self.last_vel = 0.0
        self.last_time = None
        self.noise_seed = random.randint(0, 10000)  # Random seed for consistent noise pattern
        
    def generate(self, t, amplitude, frequency, phase, duration=None):
        """Generate a filtered noise point at time t."""
        # Initialize if first call
        if self.last_time is None:
            self.last_time = t
            self.last_pos = 0.0
            self.last_vel = 0.0
            random.seed(self.noise_seed + int(phase * 1000))  # Use phase to vary the noise pattern
        
        # Time step (dt) since last call
        dt = t - self.last_time
        if dt <= 0:
            dt = 0.01  # Fallback for first call or time resets
        
        # Filter constant based on frequency (higher frequency = faster changes)
        # This creates a low-pass filtered noise with frequency determining the cutoff
        filter_constant = min(1.0, frequency * dt * 2)
        
        # Generate random target position within amplitude range
        # The random walk is biased toward zero to prevent unbounded drift
        target_pos = self.last_pos * (1 - filter_constant * 0.1) + random.uniform(-1, 1) * amplitude * filter_constant
        
        # Limit to amplitude range
        target_pos = max(-amplitude, min(amplitude, target_pos))
        
        # Calculate position, velocity, and acceleration
        pos = target_pos
        
        # Velocity is the change in position over time
        vel = (pos - self.last_pos) / dt
        
        # Acceleration is the change in velocity over time
        acc = (vel - self.last_vel) / dt
        
        # Save current values for next call
        self.last_pos = pos
        self.last_vel = vel
        self.last_time = t
        
        return pos, vel, acc


def get_wave_generator(waveform_type):
    """
    Factory function to get the appropriate wave generator.
    
    Args:
        waveform_type (str or WaveformType): The type of waveform to generate
        
    Returns:
        WaveGenerator: An instance of the appropriate wave generator
    """
    if isinstance(waveform_type, str):
        waveform_type = WaveformType(waveform_type)
    
    wave_generators = {
        WaveformType.SINE: SineWaveGenerator(),
        WaveformType.TRIANGLE: TriangleWaveGenerator(),
        WaveformType.CHIRP: ChirpWaveGenerator(),
        WaveformType.GAUSSIAN: GaussianWaveGenerator(),
        WaveformType.SAWTOOTH: SawtoothWaveGenerator(),
        WaveformType.PULSE: PulseWaveGenerator(),
        WaveformType.NOISE: NoiseWaveGenerator()
    }
    
    return wave_generators.get(waveform_type, SineWaveGenerator()) 