"""
Main simulation class that integrates all components.

Provides unified API for running robot simulations with sensors.
Implements Section 10 (Simulation Control & Time Management).
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple

from amrx.robot.robot import Robot
from amrx.world.world import World
from amrx.sensors.lidar import LidarSensor
from amrx.sensors.odometry import OdometrySensor
from amrx.sensors.landmark_detector import LandmarkDetector


class Simulation:
    """
    Main simulation controller.

    Integrates robot, world, and sensors with time management.
    Supports real-time and fast-as-possible modes.

    Time stepping modes (Section 10.1):
    - Real-time: Target 10 Hz (100ms per frame) with visualization
    - Fast-as-possible: No sleep, headless mode for batch experiments

    Deterministic behavior (Section 10.2):
    - Same random seed → identical simulation results
    """

    def __init__(
        self,
        world: World,
        dt: float = 0.1,
        random_seed: Optional[int] = None,
        enable_odometry: bool = True,
        enable_lidar: bool = True,
        enable_landmark_detector: bool = True,
    ):
        """
        Initialize simulation.

        Args:
            world: World environment
            dt: Simulation timestep (seconds), default 0.1s = 10Hz
            random_seed: Random seed for deterministic behavior
            enable_odometry: Enable odometry sensor
            enable_lidar: Enable LiDAR sensor
            enable_landmark_detector: Enable landmark detector
        """
        self.world = world
        self.dt = dt

        # Set random seed for deterministic behavior
        if random_seed is not None:
            np.random.seed(random_seed)
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()

        # Initialize robot at world's start position
        x, y, theta = world.robot_start_pose
        self.robot = Robot(x=x, y=y, theta=theta)

        # Initialize sensors
        self.odometry = OdometrySensor(random_state=self.rng) if enable_odometry else None
        self.lidar = LidarSensor(random_state=self.rng) if enable_lidar else None
        self.landmark_detector = (
            LandmarkDetector(random_state=self.rng) if enable_landmark_detector else None
        )

        # Simulation state
        self.time = 0.0
        self.step_count = 0

        # For odometry calculations
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta

    def step(self, v_cmd: float, omega_cmd: float) -> Dict[str, Any]:
        """
        Execute one simulation timestep.

        Algorithm:
        1. Apply commanded velocities to robot (with dynamics)
        2. Check for collisions, stop if collision occurs
        3. Compute sensor measurements
        4. Return sensor data

        Args:
            v_cmd: Commanded linear velocity (m/s)
            omega_cmd: Commanded angular velocity (rad/s)

        Returns:
            Dictionary with sensor data:
            {
                'odometry': (dx_robot, dy_robot, dtheta) or None,
                'lidar': ranges array or None,
                'landmarks': list of (range, bearing, signature) or None,
                'ground_truth': (x, y, theta, v, omega),
                'collision': bool,
                'time': float,
            }
        """
        # Store previous state for odometry
        prev_state = self.robot.get_state()
        prev_x, prev_y, prev_theta, prev_v, prev_omega = prev_state

        # Update robot kinematics
        self.robot.update_kinematics(v_cmd, omega_cmd, self.dt)

        # Get new state
        curr_x, curr_y, curr_theta, curr_v, curr_omega = self.robot.get_state()

        # Check for collisions
        robot_pos = np.array([curr_x, curr_y])
        collision, obj_type = self.world.check_robot_collision(
            robot_pos, self.robot.collision_radius
        )

        # If collision, revert to previous state and stop
        if collision:
            self.robot.x = prev_x
            self.robot.y = prev_y
            self.robot.theta = prev_theta
            self.robot.v = 0.0
            self.robot.omega = 0.0
            curr_x, curr_y, curr_theta = prev_x, prev_y, prev_theta
            curr_v, curr_omega = 0.0, 0.0

        # Compute odometry measurement
        odometry_data = None
        if self.odometry is not None:
            # Calculate true motion in global frame
            dx_global = curr_x - self.prev_x
            dy_global = curr_y - self.prev_y
            dtheta = curr_theta - self.prev_theta

            # Convert to distance and rotation
            delta_s = np.sqrt(dx_global**2 + dy_global**2)
            # Handle direction (forward vs backward)
            heading_to_goal = np.arctan2(dy_global, dx_global)
            from amrx.utils.geometry import normalize_angle

            angle_diff = normalize_angle(heading_to_goal - self.prev_theta)
            if abs(angle_diff) > np.pi / 2:
                delta_s = -delta_s  # Moving backwards

            # Get noisy odometry measurement in robot frame
            odometry_data = self.odometry.measure(delta_s, dtheta)

        # Update previous state for next odometry calculation
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_theta = curr_theta

        # Compute LiDAR scan
        lidar_data = None
        if self.lidar is not None:
            lidar_data = self.lidar.scan(curr_x, curr_y, curr_theta, self.world)

        # Detect landmarks
        landmark_data = None
        if self.landmark_detector is not None:
            landmark_data = self.landmark_detector.detect(curr_x, curr_y, curr_theta, self.world)

        # Update simulation time
        self.time += self.dt
        self.step_count += 1

        # Return sensor data
        return {
            "odometry": odometry_data,
            "lidar": lidar_data,
            "landmarks": landmark_data,
            "ground_truth": (curr_x, curr_y, curr_theta, curr_v, curr_omega),
            "collision": collision,
            "time": self.time,
        }

    def reset(self, x: Optional[float] = None, y: Optional[float] = None, theta: Optional[float] = None):
        """
        Reset simulation to initial or specified state.

        Args:
            x: X position (uses world start if None)
            y: Y position (uses world start if None)
            theta: Heading angle (uses world start if None)
        """
        # Use world start pose if not specified
        if x is None or y is None or theta is None:
            x, y, theta = self.world.robot_start_pose

        # Reset robot
        self.robot.reset(x, y, theta)

        # Reset simulation state
        self.time = 0.0
        self.step_count = 0

        # Reset previous state for odometry
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta

    def get_robot_pose(self) -> Tuple[float, float, float]:
        """
        Get current robot pose (ground truth).

        For evaluation/visualization only.
        Students shouldn't call this in their algorithms.

        Returns:
            (x, y, theta) tuple
        """
        return self.robot.get_pose()

    def get_robot_state(self) -> Tuple[float, float, float, float, float]:
        """
        Get current robot state (ground truth).

        For evaluation/visualization only.

        Returns:
            (x, y, theta, v, omega) tuple
        """
        return self.robot.get_state()

    def run_open_loop(
        self,
        v_cmd: float,
        omega_cmd: float,
        duration: float,
        return_history: bool = False,
    ) -> Optional[Dict[str, list]]:
        """
        Run open-loop control for specified duration.

        Useful for testing and simple trajectories.

        Args:
            v_cmd: Constant linear velocity command (m/s)
            omega_cmd: Constant angular velocity command (rad/s)
            duration: Duration to run (seconds)
            return_history: If True, return history of states and sensor data

        Returns:
            If return_history=True: Dictionary with lists of states and sensor data
            Otherwise: None
        """
        num_steps = int(duration / self.dt)
        history = {
            "time": [],
            "ground_truth": [],
            "odometry": [],
            "lidar": [],
            "landmarks": [],
            "collision": [],
        } if return_history else None

        for _ in range(num_steps):
            data = self.step(v_cmd, omega_cmd)

            if return_history:
                history["time"].append(data["time"])
                history["ground_truth"].append(data["ground_truth"])
                history["odometry"].append(data["odometry"])
                history["lidar"].append(data["lidar"])
                history["landmarks"].append(data["landmarks"])
                history["collision"].append(data["collision"])

            # Stop if collision
            if data["collision"]:
                break

        return history

    def __repr__(self) -> str:
        return (
            f"Simulation(world={self.world.name}, dt={self.dt}s, "
            f"time={self.time:.2f}s, steps={self.step_count})"
        )
