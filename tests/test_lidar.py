"""
Unit tests for LiDAR sensor.

Tests from Section 11.2 (Integration Tests) and Section 4 (LiDAR spec).
"""

import pytest
import numpy as np

from amrx.sensors.lidar import LidarSensor
from amrx.world.world import World


class TestLidarInitialization:
    """Test LiDAR sensor initialization."""

    def test_default_initialization(self):
        """Test: Default LiDAR matches SICK LMS200 spec."""
        lidar = LidarSensor()

        assert np.isclose(lidar.fov, np.pi)  # 180 degrees
        assert lidar.num_rays == 361
        assert lidar.max_range == 8.0
        assert lidar.min_range == 0.1
        assert lidar.noise_std == 0.01

    def test_custom_initialization(self):
        """Test: Custom LiDAR parameters."""
        lidar = LidarSensor(
            fov=np.pi / 2,  # 90 degrees
            max_range=10.0,
            noise_std=0.02,
        )

        assert np.isclose(lidar.fov, np.pi / 2)
        assert lidar.max_range == 10.0
        assert lidar.noise_std == 0.02

    def test_ray_angles_range(self):
        """Test: Ray angles span from -90° to +90°."""
        lidar = LidarSensor()

        assert np.isclose(lidar.ray_angles[0], -np.pi / 2)  # Right side
        assert np.isclose(lidar.ray_angles[180], 0.0)  # Forward
        assert np.isclose(lidar.ray_angles[360], np.pi / 2)  # Left side


class TestLidarScanning:
    """Test LiDAR scanning in different environments."""

    def test_scan_empty_box_center(self):
        """
        Test from spec 11.2: LiDAR in center of 10m box.
        All rays (except corners) should return ~5m.
        """
        world = World.from_json("maps/empty_box.json")
        lidar = LidarSensor(noise_std=0.0)  # No noise for deterministic test

        # Robot at center, facing forward
        ranges = lidar.scan(5.0, 5.0, 0.0, world)

        assert len(ranges) == 361

        # Forward ray (index 180) should hit wall at ~5m distance
        # (from center to wall)
        forward_range = ranges[180]
        assert 4.8 < forward_range < 5.2  # Allow small numerical error

        # Most rays should hit walls around 5m
        # (excluding corners which are farther)
        median_range = np.median(ranges)
        assert 4.5 < median_range < 5.5

    def test_scan_near_wall(self):
        """Test: LiDAR near wall should detect close obstacle."""
        world = World.from_json("maps/empty_box.json")
        lidar = LidarSensor(noise_std=0.0)

        # Robot very close to bottom wall (y=0)
        ranges = lidar.scan(5.0, 0.5, 0.0, world)

        # Backward rays (index ~0-90) should detect very close wall
        backward_range = ranges[0]  # Pointing right-backward
        assert backward_range < 1.0

    def test_scan_detects_landmarks(self):
        """Test: LiDAR scan completes with landmarks present."""
        world = World.from_json("maps/empty_box.json")
        lidar = LidarSensor(noise_std=0.0)

        # Robot at (2, 2), there's a landmark nearby at (1, 1)
        ranges = lidar.scan(2.0, 2.0, 0.0, world)

        # Basic validation: scan returns correct shape and valid ranges
        assert len(ranges) == 361
        assert np.all(ranges >= lidar.min_range)
        assert np.all(ranges <= lidar.max_range)
        # At (2,2), walls are at minimum 2m away
        assert np.min(ranges) <= 2.5  # Some rays hit walls or landmarks

    def test_scan_max_range_clipping(self):
        """Test: Ranges are clipped to max_range."""
        world = World.from_json("maps/empty_box.json")
        lidar = LidarSensor(max_range=3.0, noise_std=0.0)

        # Robot at center of 10m box, so walls are 5m away
        ranges = lidar.scan(5.0, 5.0, 0.0, world)

        # All ranges should be ≤ max_range
        assert np.all(ranges <= 3.0)

        # Most should be exactly max_range (walls too far)
        assert np.sum(np.isclose(ranges, 3.0)) > 300

    def test_scan_min_range_clipping(self):
        """Test: Ranges are clipped to min_range."""
        world = World.from_json("maps/empty_box.json")
        lidar = LidarSensor(min_range=1.0, noise_std=0.0)

        # Robot very close to wall
        ranges = lidar.scan(5.0, 0.2, 0.0, world)

        # All ranges should be ≥ min_range
        assert np.all(ranges >= 1.0)


class TestLidarNoise:
    """Test LiDAR measurement noise."""

    def test_noise_reproducibility(self):
        """Test: Same random seed produces identical scans."""
        world = World.from_json("maps/empty_box.json")

        rng1 = np.random.RandomState(42)
        lidar1 = LidarSensor(random_state=rng1)
        ranges1 = lidar1.scan(5.0, 5.0, 0.0, world)

        rng2 = np.random.RandomState(42)
        lidar2 = LidarSensor(random_state=rng2)
        ranges2 = lidar2.scan(5.0, 5.0, 0.0, world)

        assert np.allclose(ranges1, ranges2)

    def test_noise_adds_variability(self):
        """Test: Noise creates variation in measurements."""
        world = World.from_json("maps/empty_box.json")

        lidar_noisy = LidarSensor(noise_std=0.1)
        ranges1 = lidar_noisy.scan(5.0, 5.0, 0.0, world)
        ranges2 = lidar_noisy.scan(5.0, 5.0, 0.0, world)

        # Different scans should differ (due to noise)
        assert not np.allclose(ranges1, ranges2)

    def test_noise_magnitude(self):
        """Test: Noise has expected standard deviation."""
        world = World.from_json("maps/empty_box.json")
        lidar = LidarSensor(noise_std=0.05)

        # Take multiple scans
        num_scans = 100
        all_ranges = np.array([lidar.scan(5.0, 5.0, 0.0, world) for _ in range(num_scans)])

        # Check standard deviation of forward ray
        forward_std = np.std(all_ranges[:, 180])
        # Should be close to 0.05, but allow some statistical variation
        assert 0.03 < forward_std < 0.07


class TestLidarUtilities:
    """Test LiDAR utility functions."""

    def test_get_ray_endpoints(self):
        """Test: Convert ranges to endpoint coordinates."""
        lidar = LidarSensor()
        ranges = np.full(361, 1.0)  # All rays at 1m

        endpoints = lidar.get_ray_endpoints(0.0, 0.0, 0.0, ranges)

        assert endpoints.shape == (361, 2)

        # Forward ray (index 180) should be at (1, 0)
        assert np.allclose(endpoints[180], [1.0, 0.0], atol=1e-6)

        # Left ray (index 360, +90°) should be at (0, 1)
        assert np.allclose(endpoints[360], [0.0, 1.0], atol=1e-6)

        # Right ray (index 0, -90°) should be at (0, -1)
        assert np.allclose(endpoints[0], [0.0, -1.0], atol=1e-6)

    def test_get_ray_angles_global(self):
        """Test: Convert sensor angles to global frame."""
        lidar = LidarSensor()

        # Robot facing east (θ=0)
        angles = lidar.get_ray_angles_global(0.0)
        assert np.isclose(angles[180], 0.0)  # Forward is east

        # Robot facing north (θ=π/2)
        angles = lidar.get_ray_angles_global(np.pi / 2)
        assert np.isclose(angles[180], np.pi / 2)  # Forward is north

    def test_repr(self):
        """Test: String representation."""
        lidar = LidarSensor()
        repr_str = repr(lidar)

        assert "LidarSensor" in repr_str
        assert "180" in repr_str  # FOV in degrees
        assert "361" in repr_str  # Number of rays


class TestLidarInComplexEnvironment:
    """Test LiDAR in SLAM arena with internal walls."""

    def test_scan_slam_arena(self):
        """Test: LiDAR in SLAM arena with rooms."""
        world = World.from_json("maps/slam_arena.json")
        lidar = LidarSensor(noise_std=0.0)

        # Robot at (1, 1)
        ranges = lidar.scan(1.0, 1.0, 0.0, world)

        assert len(ranges) == 361
        assert np.all(ranges >= lidar.min_range)
        assert np.all(ranges <= lidar.max_range)

        # Should detect nearby walls
        min_range = np.min(ranges)
        assert min_range < 2.0  # Something close

    def test_scan_corridor(self):
        """Test: LiDAR in narrow corridor."""
        world = World.from_json("maps/corridor.json")
        lidar = LidarSensor(noise_std=0.0)

        # Robot in middle of corridor at (5, 1.5)
        ranges = lidar.scan(5.0, 1.5, 0.0, world)

        # Side rays should detect close walls (~1.5m)
        # Left and right should be similar (symmetric corridor)
        left_range = ranges[360]  # +90 degrees
        right_range = ranges[0]  # -90 degrees

        assert 1.0 < left_range < 2.0
        assert 1.0 < right_range < 2.0
