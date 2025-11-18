"""
Landmark detector sensor model.

Simulates a "perfect" feature extractor for EKF-SLAM exercises.
Abstracts away vision processing and feature matching.

Implements Section 5 of specification.
"""

import numpy as np
from typing import List, Tuple, Optional

from amrx.world.world import World
from amrx.world.geometry import Landmark
from amrx.utils.geometry import normalize_angle, ray_line_intersection


class LandmarkDetector:
    """
    Landmark detection sensor for SLAM.

    This simulates a vision system that can identify and localize
    cylindrical landmarks with unique signatures.

    Detection criteria (Section 5.2):
    - Distance: d < detection_range (default 5.0m)
    - Bearing: |φ| < FOV/2 (default ±45°, 90° total)
    - Line-of-sight: No walls between robot and landmark

    Measurement format (Section 5.3):
    For each detected landmark j:
    - r_j: Range (meters) with noise N(0, σ_r²)
    - φ_j: Bearing angle (radians, robot frame) with noise N(0, σ_φ²)
    - s_j: Signature ID (integer, deterministic)
    """

    def __init__(
        self,
        detection_range: float = 5.0,
        field_of_view: float = np.pi / 2,  # 90 degrees
        range_noise_std: float = 0.05,  # 5cm
        bearing_noise_std: float = 0.02,  # ~1.1 degrees
        random_state: Optional[np.random.RandomState] = None,
    ):
        """
        Initialize landmark detector.

        Args:
            detection_range: Maximum detection distance (meters)
            field_of_view: Total FOV (radians), centered forward
            range_noise_std: Standard deviation of range noise (meters)
            bearing_noise_std: Standard deviation of bearing noise (radians)
            random_state: Random state for reproducibility
        """
        self.detection_range = detection_range
        self.fov = field_of_view
        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std

        # Random state for noise
        self.rng = random_state if random_state is not None else np.random.RandomState()

    def detect(
        self, robot_x: float, robot_y: float, robot_theta: float, world: World
    ) -> List[Tuple[float, float, int]]:
        """
        Detect visible landmarks.

        Algorithm (Section 5.2):
        1. For each landmark in world:
           a. Calculate distance and bearing
           b. Check if within detection range
           c. Check if within FOV
           d. Check line-of-sight (no wall occlusion)
        2. Add measurement noise
        3. Return sorted by range (closest first)

        Args:
            robot_x: Robot X position (meters)
            robot_y: Robot Y position (meters)
            robot_theta: Robot heading angle (radians)
            world: World environment

        Returns:
            List of (range, bearing, signature) tuples
            - range: Distance to landmark (meters)
            - bearing: Angle in robot frame (radians)
            - signature: Landmark unique ID
        """
        robot_pos = np.array([robot_x, robot_y])
        detections = []

        for landmark in world.landmarks:
            # Calculate true range and bearing
            dx = landmark.center[0] - robot_x
            dy = landmark.center[1] - robot_y
            range_true = np.sqrt(dx * dx + dy * dy)

            # Check distance criterion
            if range_true > self.detection_range:
                continue

            # Calculate bearing in global frame, then convert to robot frame
            bearing_global = np.arctan2(dy, dx)
            bearing_robot = normalize_angle(bearing_global - robot_theta)

            # Check FOV criterion
            if abs(bearing_robot) > self.fov / 2:
                continue

            # Check line-of-sight (occlusion by walls)
            if self._is_occluded(robot_pos, landmark, world):
                continue

            # Add measurement noise
            range_measured = range_true + self.rng.normal(0.0, self.range_noise_std)
            bearing_measured = bearing_robot + self.rng.normal(
                0.0, self.bearing_noise_std
            )

            # Clip range to positive values
            range_measured = max(0.0, range_measured)

            # Normalize bearing
            bearing_measured = normalize_angle(bearing_measured)

            detections.append((range_measured, bearing_measured, landmark.signature))

        # Sort by range (closest first) for consistency
        detections.sort(key=lambda x: x[0])

        return detections

    def _is_occluded(
        self, robot_pos: np.ndarray, landmark: Landmark, world: World
    ) -> bool:
        """
        Check if landmark is occluded by walls.

        Casts a ray from robot to landmark center and checks if any
        wall intersects before reaching the landmark.

        Args:
            robot_pos: Robot position (x, y)
            landmark: Landmark to check
            world: World environment

        Returns:
            True if landmark is occluded, False otherwise
        """
        # Ray from robot to landmark
        direction = landmark.center - robot_pos
        distance_to_landmark = np.linalg.norm(direction)

        if distance_to_landmark < 1e-6:
            return False  # Robot on top of landmark

        direction = direction / distance_to_landmark  # Normalize

        # Check all walls
        for wall in world.walls:
            intersects, t = ray_line_intersection(
                robot_pos, direction, wall.start, wall.end
            )

            # If ray hits wall before reaching landmark, it's occluded
            if intersects and t is not None and t < distance_to_landmark:
                return True

        return False

    def get_landmark_global_position(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        range_measured: float,
        bearing_measured: float,
    ) -> Tuple[float, float]:
        """
        Convert landmark measurement to global coordinates.

        Useful for SLAM algorithms to initialize landmark positions.

        Args:
            robot_x: Robot X position (meters)
            robot_y: Robot Y position (meters)
            robot_theta: Robot heading (radians)
            range_measured: Measured range to landmark (meters)
            bearing_measured: Measured bearing in robot frame (radians)

        Returns:
            (x, y) global position estimate
        """
        # Bearing in global frame
        bearing_global = robot_theta + bearing_measured

        # Global position
        x = robot_x + range_measured * np.cos(bearing_global)
        y = robot_y + range_measured * np.sin(bearing_global)

        return x, y

    def __repr__(self) -> str:
        return (
            f"LandmarkDetector(range={self.detection_range}m, "
            f"fov={np.degrees(self.fov):.0f}°, "
            f"σ_r={self.range_noise_std}m, "
            f"σ_φ={self.bearing_noise_std:.3f}rad)"
        )
