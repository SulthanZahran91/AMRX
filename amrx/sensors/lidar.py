"""
2D LiDAR sensor model.

Implements SICK LMS200 profile as per Section 4 of specification:
- 180° field of view
- 0.5° angular resolution (361 rays)
- 8m max range
- Gaussian measurement noise
"""

import numpy as np
from typing import Optional

from amrx.world.world import World
from amrx.utils.geometry import ray_line_intersection, ray_circle_intersection


class LidarSensor:
    """
    2D LiDAR sensor with realistic noise model.

    Specification (Section 4.1):
    - Field of View: 180° (π radians)
    - Angular Resolution: 0.5° (0.00873 radians) → 361 rays
    - Max Range: 8.0 m
    - Min Range: 0.1 m
    - Measurement Noise: σ_r = 0.01 m (Gaussian)

    Ray indexing:
    - rays[0]: -90° (robot's right side)
    - rays[180]: 0° (forward)
    - rays[360]: +90° (robot's left side)
    """

    def __init__(
        self,
        fov: float = np.pi,  # 180 degrees
        angular_resolution: float = 0.00873,  # 0.5 degrees
        max_range: float = 8.0,
        min_range: float = 0.1,
        noise_std: float = 0.01,
        random_state: Optional[np.random.RandomState] = None,
    ):
        """
        Initialize LiDAR sensor.

        Args:
            fov: Field of view (radians)
            angular_resolution: Angular resolution (radians)
            max_range: Maximum sensing range (meters)
            min_range: Minimum sensing range (meters)
            noise_std: Standard deviation of range noise (meters)
            random_state: Random state for reproducibility
        """
        self.fov = fov
        self.angular_resolution = angular_resolution
        self.max_range = max_range
        self.min_range = min_range
        self.noise_std = noise_std

        # Calculate number of rays
        # Use round to handle floating point precision, then add 1 for endpoints
        self.num_rays = int(round(fov / angular_resolution)) + 1  # 361 rays for default

        # Precompute ray angles in sensor frame
        # From -π/2 to +π/2
        self.ray_angles = np.linspace(
            -fov / 2, fov / 2, self.num_rays, dtype=np.float64
        )

        # Random state for noise
        self.rng = random_state if random_state is not None else np.random.RandomState()

    def scan(
        self, robot_x: float, robot_y: float, robot_theta: float, world: World
    ) -> np.ndarray:
        """
        Perform a 2D LiDAR scan.

        Algorithm (Section 4.3):
        1. For each ray, compute global angle: θ_ray = θ_robot + φ_i
        2. Cast ray from robot position
        3. Find minimum intersection distance with all obstacles
        4. Add measurement noise
        5. Clip to [min_range, max_range]

        Args:
            robot_x: Robot X position (meters)
            robot_y: Robot Y position (meters)
            robot_theta: Robot heading angle (radians)
            world: World environment

        Returns:
            Array of ranges (361,) in meters
        """
        ranges = np.full(self.num_rays, self.max_range, dtype=np.float64)
        robot_pos = np.array([robot_x, robot_y])

        for i, phi in enumerate(self.ray_angles):
            # Global ray angle
            theta_ray = robot_theta + phi

            # Ray direction (unit vector)
            direction = np.array([np.cos(theta_ray), np.sin(theta_ray)])

            # Find closest intersection
            min_distance = self.max_range

            # Check walls
            for wall in world.walls:
                intersects, t = ray_line_intersection(
                    robot_pos, direction, wall.start, wall.end
                )
                if intersects and t is not None:
                    min_distance = min(min_distance, t)

            # Check landmarks (cylinders)
            for landmark in world.landmarks:
                intersects, t = ray_circle_intersection(
                    robot_pos, direction, landmark.center, landmark.radius
                )
                if intersects and t is not None:
                    min_distance = min(min_distance, t)

            ranges[i] = min_distance

        # Add measurement noise (Gaussian)
        noise = self.rng.normal(0.0, self.noise_std, size=self.num_rays)
        ranges += noise

        # Clip to valid range
        ranges = np.clip(ranges, self.min_range, self.max_range)

        return ranges

    def get_ray_endpoints(
        self, robot_x: float, robot_y: float, robot_theta: float, ranges: np.ndarray
    ) -> np.ndarray:
        """
        Convert range measurements to endpoint coordinates.

        Useful for visualization.

        Args:
            robot_x: Robot X position (meters)
            robot_y: Robot Y position (meters)
            robot_theta: Robot heading angle (radians)
            ranges: Range measurements (361,)

        Returns:
            Array of endpoints (361, 2) with (x, y) coordinates
        """
        endpoints = np.zeros((self.num_rays, 2))

        for i, (phi, r) in enumerate(zip(self.ray_angles, ranges)):
            theta_ray = robot_theta + phi
            endpoints[i, 0] = robot_x + r * np.cos(theta_ray)
            endpoints[i, 1] = robot_y + r * np.sin(theta_ray)

        return endpoints

    def get_ray_angles_global(self, robot_theta: float) -> np.ndarray:
        """
        Get ray angles in global frame.

        Args:
            robot_theta: Robot heading angle (radians)

        Returns:
            Array of global ray angles (361,)
        """
        return robot_theta + self.ray_angles

    def __repr__(self) -> str:
        return (
            f"LidarSensor(fov={np.degrees(self.fov):.1f}°, "
            f"num_rays={self.num_rays}, "
            f"max_range={self.max_range}m)"
        )
