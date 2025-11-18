"""
Robot physics and kinematics module.

Implements differential drive robot model with:
- Kinematic state (position, orientation, velocities)
- Differential drive equations
- Wheel speed relationships
"""

import numpy as np
from typing import Tuple
from amrx.utils.geometry import normalize_angle


class Robot:
    """
    Differential drive robot with realistic kinematics.

    State vector (ground truth):
        x_true = [x, y, θ, v, ω]^T

    Coordinates:
        - X-axis: East (right)
        - Y-axis: North (up)
        - θ = 0: Points along +X axis
        - Angular direction: Counter-clockwise positive

    Parameters (Pioneer P3-DX):
        - Wheel base: b = 0.381 m
        - Wheel radius: r = 0.0975 m
        - Collision radius: R = 0.2 m
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        wheel_base: float = 0.381,
        wheel_radius: float = 0.0975,
        collision_radius: float = 0.2,
        max_linear_velocity: float = 1.2,
        max_angular_velocity: float = 1.745,
        max_linear_accel: float = 1.0,
        max_angular_accel: float = 2.0,
    ):
        """
        Initialize robot at given pose.

        Args:
            x: Initial X position (meters)
            y: Initial Y position (meters)
            theta: Initial heading angle (radians)
            wheel_base: Distance between wheels (meters)
            wheel_radius: Radius of each wheel (meters)
            collision_radius: Robot bounding circle radius (meters)
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            max_linear_accel: Maximum linear acceleration (m/s²)
            max_angular_accel: Maximum angular acceleration (rad/s²)
        """
        # State variables
        self.x = x
        self.y = y
        self.theta = normalize_angle(theta)
        self.v = 0.0  # Linear velocity (m/s)
        self.omega = 0.0  # Angular velocity (rad/s)

        # Robot parameters
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.collision_radius = collision_radius

        # Dynamic constraints (Section 2.4)
        self.v_max = max_linear_velocity
        self.omega_max = max_angular_velocity
        self.a_max = max_linear_accel
        self.alpha_max = max_angular_accel

        # Store previous position for trail visualization
        self.trail = [(self.x, self.y)]

    def update_kinematics(self, v_cmd: float, omega_cmd: float, dt: float) -> None:
        """
        Update robot state using differential drive kinematics with dynamics.

        Uses Euler integration (as per spec Section 2.2):
            x_{k+1} = x_k + v_k * cos(θ_k) * Δt
            y_{k+1} = y_k + v_k * sin(θ_k) * Δt
            θ_{k+1} = normalize(θ_k + ω_k * Δt)

        Implementation order (Section 2.4):
        1. Clip command to max velocities
        2. Apply slew rate limit (acceleration constraints)
        3. Update kinematics with achieved velocity (not commanded)

        Args:
            v_cmd: Commanded linear velocity (m/s)
            omega_cmd: Commanded angular velocity (rad/s)
            dt: Time step (seconds)
        """
        # Step 1: Clip to maximum velocities
        v_cmd = np.clip(v_cmd, -self.v_max, self.v_max)
        omega_cmd = np.clip(omega_cmd, -self.omega_max, self.omega_max)

        # Step 2: Apply slew rate limiting (acceleration constraints)
        # Linear velocity
        dv_max = self.a_max * dt  # Maximum velocity change this timestep
        dv = np.clip(v_cmd - self.v, -dv_max, dv_max)
        self.v += dv

        # Angular velocity
        domega_max = self.alpha_max * dt  # Maximum angular velocity change
        domega = np.clip(omega_cmd - self.omega, -domega_max, domega_max)
        self.omega += domega

        # Step 3: Update kinematics with achieved velocity
        # Euler integration
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Normalize angle to [-π, π]
        self.theta = normalize_angle(self.theta)

        # Update trail for visualization
        self.trail.append((self.x, self.y))
        if len(self.trail) > 100:  # Keep last 100 positions
            self.trail.pop(0)

    def get_state(self) -> Tuple[float, float, float, float, float]:
        """
        Get current robot state (ground truth).

        Returns:
            (x, y, theta, v, omega) tuple
        """
        return self.x, self.y, self.theta, self.v, self.omega

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Get current robot pose.

        Returns:
            (x, y, theta) tuple
        """
        return self.x, self.y, self.theta

    def reset(self, x: float, y: float, theta: float) -> None:
        """
        Teleport robot to new pose.

        Used for resetting between trials.

        Args:
            x: New X position (meters)
            y: New Y position (meters)
            theta: New heading angle (radians)
        """
        self.x = x
        self.y = y
        self.theta = normalize_angle(theta)
        self.v = 0.0
        self.omega = 0.0
        self.trail = [(self.x, self.y)]

    def wheel_speeds_to_body_velocity(
        self, omega_left: float, omega_right: float
    ) -> Tuple[float, float]:
        """
        Forward kinematics: Convert wheel speeds to body velocities.

        From Section 2.3:
            v = r(ω_R + ω_L) / 2
            ω = r(ω_R - ω_L) / b

        Args:
            omega_left: Left wheel angular velocity (rad/s)
            omega_right: Right wheel angular velocity (rad/s)

        Returns:
            (v, omega) - linear and angular velocities
        """
        v = self.wheel_radius * (omega_right + omega_left) / 2.0
        omega = self.wheel_radius * (omega_right - omega_left) / self.wheel_base
        return v, omega

    def body_velocity_to_wheel_speeds(
        self, v: float, omega: float
    ) -> Tuple[float, float]:
        """
        Inverse kinematics: Convert body velocities to wheel speeds.

        From Section 2.3:
            ω_L = (v - ωb/2) / r
            ω_R = (v + ωb/2) / r

        Args:
            v: Linear velocity (m/s)
            omega: Angular velocity (rad/s)

        Returns:
            (omega_left, omega_right) - wheel angular velocities (rad/s)
        """
        omega_left = (v - omega * self.wheel_base / 2.0) / self.wheel_radius
        omega_right = (v + omega * self.wheel_base / 2.0) / self.wheel_radius
        return omega_left, omega_right

    def get_vertices(self) -> np.ndarray:
        """
        Get robot body vertices for visualization (triangle).

        Returns robot as a triangle pointing in the heading direction.
        Triangle size based on collision radius.

        Returns:
            Array of shape (3, 2) with triangle vertices
        """
        # Triangle dimensions relative to heading
        length = self.collision_radius * 1.5  # Forward point
        width = self.collision_radius  # Side points

        # Vertices in robot frame
        vertices_local = np.array([
            [length, 0],  # Front point
            [-width / 2, width / 2],  # Back left
            [-width / 2, -width / 2],  # Back right
        ])

        # Rotation matrix
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Transform to global frame
        vertices_global = vertices_local @ rotation.T
        vertices_global[:, 0] += self.x
        vertices_global[:, 1] += self.y

        return vertices_global
