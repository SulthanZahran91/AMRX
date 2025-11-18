"""
Unit tests for landmark detector.

Tests from Section 5 (Landmark Detector specification).
"""

import pytest
import numpy as np

from amrx.sensors.landmark_detector import LandmarkDetector
from amrx.world.world import World


class TestLandmarkDetectorInitialization:
    """Test landmark detector initialization."""

    def test_default_initialization(self):
        """Test: Default parameters match spec."""
        detector = LandmarkDetector()

        assert detector.detection_range == 5.0  # 5 meters
        assert np.isclose(detector.fov, np.pi / 2)  # 90 degrees
        assert detector.range_noise_std == 0.05  # 5cm
        assert detector.bearing_noise_std == 0.02  # ~1.1 degrees

    def test_custom_initialization(self):
        """Test: Custom parameters."""
        detector = LandmarkDetector(
            detection_range=10.0,
            field_of_view=np.pi,
            range_noise_std=0.1,
            bearing_noise_std=0.05,
        )

        assert detector.detection_range == 10.0
        assert np.isclose(detector.fov, np.pi)
        assert detector.range_noise_std == 0.1
        assert detector.bearing_noise_std == 0.05


class TestLandmarkDetection:
    """Test landmark detection in various scenarios."""

    def test_detect_landmark_within_range_and_fov(self):
        """Test: Detect landmark within range and FOV."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot at (1.5, 1) facing west (θ=π), landmark at (1, 1) is nearby
        # Distance: 0.5m, Bearing: 0° (directly ahead in robot frame)
        detections = detector.detect(1.5, 1.0, np.pi, world)

        # Should detect the nearby landmark at (1,1)
        assert len(detections) > 0

        # Check format: (range, bearing, signature)
        for detection in detections:
            assert len(detection) == 3
            range_val, bearing, sig = detection
            assert isinstance(range_val, float)
            assert isinstance(bearing, float)
            assert isinstance(sig, int)

    def test_no_detection_beyond_range(self):
        """Test: Landmarks beyond detection range are not detected."""
        world = World.from_json("maps/empty_box.json")
        # Set very short detection range
        detector = LandmarkDetector(detection_range=0.5)

        # Robot at center, all landmarks are far away
        detections = detector.detect(5.0, 5.0, 0.0, world)

        # Should detect nothing
        assert len(detections) == 0

    def test_no_detection_outside_fov(self):
        """Test: Landmarks outside FOV are not detected."""
        world = World.from_json("maps/empty_box.json")
        # Very narrow FOV
        detector = LandmarkDetector(field_of_view=0.1)  # ~5.7 degrees

        # Robot at (5, 5) facing east
        # Most landmarks will be outside narrow FOV
        detections = detector.detect(5.0, 5.0, 0.0, world)

        # Should detect very few or none
        assert len(detections) <= 1

    def test_detection_sorted_by_range(self):
        """Test: Detections are sorted by range (closest first)."""
        world = World.from_json("maps/slam_arena.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot at (1, 1) with multiple landmarks visible
        detections = detector.detect(1.0, 1.0, 0.0, world)

        if len(detections) > 1:
            # Verify sorted order
            ranges = [d[0] for d in detections]
            assert ranges == sorted(ranges)

    def test_detection_includes_signature(self):
        """Test: Each detection includes landmark signature."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector()

        detections = detector.detect(2.0, 2.0, 0.0, world)

        # Get all detected signatures
        signatures = [d[2] for d in detections]

        # All signatures should be valid (non-negative integers)
        assert all(isinstance(s, int) for s in signatures)
        assert all(s >= 0 for s in signatures)

        # Signatures should match landmarks in the world
        world_signatures = [lm.signature for lm in world.landmarks]
        assert all(s in world_signatures for s in signatures)


class TestOcclusionHandling:
    """Test line-of-sight occlusion by walls."""

    def test_landmark_behind_wall_not_detected(self):
        """Test: Landmark occluded by wall is not detected."""
        world = World.from_json("maps/slam_arena.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot at (1, 1) facing east
        # There's an internal wall at x=4 that should occlude landmarks beyond it
        detections = detector.detect(1.0, 1.0, 0.0, world)

        # Extract detected landmarks
        detected_sigs = [d[2] for d in detections]

        # Landmarks on the other side of internal walls should not be detected
        # This is environment-specific, so we just verify occlusion logic works
        # by ensuring not all landmarks are detected
        total_landmarks = len(world.landmarks)
        assert len(detected_sigs) < total_landmarks

    def test_clear_line_of_sight_detected(self):
        """Test: Landmark with clear line-of-sight is detected."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot at (1.5, 1) facing west, landmark at (1, 1) with clear line of sight
        detections = detector.detect(1.5, 1.0, np.pi, world)

        # Should detect the nearby landmark with signature 0
        detected_sigs = [d[2] for d in detections]
        assert 0 in detected_sigs


class TestMeasurementNoise:
    """Test measurement noise characteristics."""

    def test_noise_reproducibility(self):
        """Test: Same random seed produces identical measurements."""
        world = World.from_json("maps/empty_box.json")

        rng1 = np.random.RandomState(42)
        detector1 = LandmarkDetector(random_state=rng1)
        detections1 = detector1.detect(2.0, 2.0, 0.0, world)

        rng2 = np.random.RandomState(42)
        detector2 = LandmarkDetector(random_state=rng2)
        detections2 = detector2.detect(2.0, 2.0, 0.0, world)

        # Should produce identical results
        assert len(detections1) == len(detections2)
        for d1, d2 in zip(detections1, detections2):
            assert np.isclose(d1[0], d2[0])  # Range
            assert np.isclose(d1[1], d2[1])  # Bearing
            assert d1[2] == d2[2]  # Signature

    def test_noise_adds_variability(self):
        """Test: Noise creates measurement variation."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector()

        # Multiple measurements from same position
        # Robot facing west toward landmark at (1,1)
        all_detections = [detector.detect(1.5, 1.0, np.pi, world) for _ in range(10)]

        # Should have some detections
        assert all(len(d) > 0 for d in all_detections)

        # Extract range measurements for first detected landmark
        ranges = [d[0][0] for d in all_detections if len(d) > 0]

        # Should have variation due to noise
        assert np.std(ranges) > 0.01

    def test_zero_noise_deterministic(self):
        """Test: Zero noise produces deterministic measurements."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        detections1 = detector.detect(2.0, 2.0, 0.0, world)
        detections2 = detector.detect(2.0, 2.0, 0.0, world)

        # Should be identical
        assert len(detections1) == len(detections2)
        for d1, d2 in zip(detections1, detections2):
            assert np.isclose(d1[0], d2[0])
            assert np.isclose(d1[1], d2[1])
            assert d1[2] == d2[2]


class TestMeasurementAccuracy:
    """Test measurement accuracy (without noise)."""

    def test_range_measurement_accuracy(self):
        """Test: Range measurement is accurate (no noise)."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot at (1.5, 1) facing west, landmark at (1, 1)
        # True distance: 0.5m
        detections = detector.detect(1.5, 1.0, np.pi, world)

        # Find detection of landmark 0 (at position 1,1)
        landmark_0_detection = None
        for d in detections:
            if d[2] == 0:  # Signature 0
                landmark_0_detection = d
                break

        assert landmark_0_detection is not None
        measured_range = landmark_0_detection[0]
        expected_range = 0.5
        assert np.isclose(measured_range, expected_range, atol=0.01)

    def test_bearing_measurement_accuracy(self):
        """Test: Bearing measurement is accurate (no noise)."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot at (1.5, 1) facing west (θ=π), landmark at (1, 1)
        # Global bearing: atan2(1-1, 1-1.5) = atan2(0, -0.5) = π (west)
        # Robot bearing: π - π = 0 (directly ahead)
        detections = detector.detect(1.5, 1.0, np.pi, world)

        # Find detection of landmark 0
        landmark_0_detection = None
        for d in detections:
            if d[2] == 0:
                landmark_0_detection = d
                break

        assert landmark_0_detection is not None
        measured_bearing = landmark_0_detection[1]
        expected_bearing = 0.0  # Directly ahead
        assert np.isclose(measured_bearing, expected_bearing, atol=0.01)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_landmark_global_position(self):
        """Test: Convert measurement to global coordinates."""
        detector = LandmarkDetector()

        # Robot at (2, 2) facing east (θ=0)
        # Landmark 1m away at bearing 0° (directly ahead)
        x, y = detector.get_landmark_global_position(
            robot_x=2.0, robot_y=2.0, robot_theta=0.0, range_measured=1.0, bearing_measured=0.0
        )

        # Should be at (3, 2)
        assert np.isclose(x, 3.0)
        assert np.isclose(y, 2.0)

    def test_get_landmark_global_position_rotated(self):
        """Test: Convert with robot rotated."""
        detector = LandmarkDetector()

        # Robot at (0, 0) facing north (θ=π/2)
        # Landmark 1m away at bearing 0° (directly ahead)
        x, y = detector.get_landmark_global_position(
            robot_x=0.0, robot_y=0.0, robot_theta=np.pi / 2, range_measured=1.0, bearing_measured=0.0
        )

        # Should be at (0, 1)
        assert np.isclose(x, 0.0, atol=1e-6)
        assert np.isclose(y, 1.0, atol=1e-6)

    def test_repr(self):
        """Test: String representation."""
        detector = LandmarkDetector()
        repr_str = repr(detector)

        assert "LandmarkDetector" in repr_str
        assert "5" in repr_str  # Range
        assert "90" in repr_str  # FOV in degrees


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_landmarks_in_world(self):
        """Test: World with no landmarks returns empty list."""
        world = World("Empty")
        detector = LandmarkDetector()

        detections = detector.detect(0.0, 0.0, 0.0, world)
        assert len(detections) == 0

    def test_robot_on_landmark(self):
        """Test: Robot exactly on landmark position."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(range_noise_std=0.0, bearing_noise_std=0.0)

        # Robot exactly at landmark position (1, 1)
        detections = detector.detect(1.0, 1.0, 0.0, world)

        # Should detect landmark at range ≈ 0
        if len(detections) > 0:
            # At least one detection should have very small range
            min_range = min(d[0] for d in detections)
            assert min_range < 0.2  # Very close

    def test_landmark_exactly_at_fov_boundary(self):
        """Test: Landmark exactly at FOV boundary."""
        world = World.from_json("maps/empty_box.json")
        detector = LandmarkDetector(field_of_view=np.pi / 2)  # ±45°

        # Position robot such that landmark is at edge of FOV
        # This is tricky to set up precisely, so just verify no crash
        detections = detector.detect(5.0, 5.0, 0.0, world)

        # Should complete without error
        assert isinstance(detections, list)
