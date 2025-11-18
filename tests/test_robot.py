"""
Unit tests for Robot class.

Tests from Section 11.1 (Unit Tests Per Component).
"""

import pytest
import numpy as np
from amrx.robot.robot import Robot


class TestRobotInitialization:
    """Test robot initialization."""

    def test_default_initialization(self):
        """Test: Robot initializes with default values."""
        robot = Robot()
        x, y, theta, v, omega = robot.get_state()

        assert x == 0.0
        assert y == 0.0
        assert theta == 0.0
        assert v == 0.0
        assert omega == 0.0

    def test_custom_initialization(self):
        """Test: Robot initializes with custom pose."""
        robot = Robot(x=1.0, y=2.0, theta=np.pi / 2)
        x, y, theta, v, omega = robot.get_state()

        assert x == 1.0
        assert y == 2.0
        assert np.isclose(theta, np.pi / 2)
        assert v == 0.0
        assert omega == 0.0

    def test_angle_normalization_on_init(self):
        """Test: Initial angle is normalized."""
        robot = Robot(theta=7 * np.pi)
        _, _, theta, _, _ = robot.get_state()
        assert np.isclose(theta, np.pi, atol=1e-10)


class TestDifferentialDriveKinematics:
    """Test differential drive kinematic equations."""

    def test_forward_motion_zero_heading(self):
        """
        Test from spec 11.1: Forward motion at θ=0.
        Command: v=1 m/s, ω=0, Δt=1s
        Expected: Δx=1, Δy=0
        """
        robot = Robot(x=0.0, y=0.0, theta=0.0)
        robot.update_kinematics(v_cmd=1.0, omega_cmd=0.0, dt=1.0)

        x, y, theta, _, _ = robot.get_state()
        assert np.isclose(x, 1.0, atol=1e-6)
        assert np.isclose(y, 0.0, atol=1e-6)
        assert np.isclose(theta, 0.0, atol=1e-6)

    def test_forward_motion_90_degrees(self):
        """Test: Forward motion at θ=90°."""
        robot = Robot(x=0.0, y=0.0, theta=np.pi / 2)
        robot.update_kinematics(v_cmd=1.0, omega_cmd=0.0, dt=1.0)

        x, y, theta, _, _ = robot.get_state()
        assert np.isclose(x, 0.0, atol=1e-6)
        assert np.isclose(y, 1.0, atol=1e-6)
        assert np.isclose(theta, np.pi / 2, atol=1e-6)

    def test_pure_rotation(self):
        """Test: Pure rotation (v=0, ω≠0)."""
        robot = Robot(x=0.0, y=0.0, theta=0.0)
        robot.update_kinematics(v_cmd=0.0, omega_cmd=np.pi / 2, dt=1.0)

        x, y, theta, _, _ = robot.get_state()
        assert np.isclose(x, 0.0, atol=1e-6)
        assert np.isclose(y, 0.0, atol=1e-6)
        assert np.isclose(theta, np.pi / 2, atol=1e-6)

    def test_curved_motion(self):
        """Test: Simultaneous linear and angular motion."""
        robot = Robot(x=0.0, y=0.0, theta=0.0)
        # Small timestep for Euler accuracy
        dt = 0.1
        for _ in range(10):
            robot.update_kinematics(v_cmd=1.0, omega_cmd=1.0, dt=dt)

        x, y, theta, _, _ = robot.get_state()
        # After 1 second at v=1, ω=1: should trace a circle
        # Rough check: position should have moved
        assert abs(x) > 0.1
        assert abs(y) > 0.1
        assert np.isclose(theta, 1.0, atol=1e-6)  # 1 radian rotation

    def test_angle_wrapping(self):
        """Test: Angle normalizes after large rotation."""
        robot = Robot(x=0.0, y=0.0, theta=0.0)
        # Rotate more than 2π
        robot.update_kinematics(v_cmd=0.0, omega_cmd=10 * np.pi, dt=1.0)

        _, _, theta, _, _ = robot.get_state()
        assert -np.pi <= theta <= np.pi


class TestWheelSpeedConversions:
    """Test forward and inverse kinematics."""

    def test_forward_kinematics_straight(self):
        """Test: Equal wheel speeds → straight motion."""
        robot = Robot()
        v, omega = robot.wheel_speeds_to_body_velocity(
            omega_left=10.0, omega_right=10.0
        )

        # Equal speeds → no rotation
        assert np.isclose(omega, 0.0, atol=1e-6)
        # Linear velocity should be positive
        assert v > 0

    def test_forward_kinematics_rotation(self):
        """Test: Opposite wheel speeds → pure rotation."""
        robot = Robot()
        v, omega = robot.wheel_speeds_to_body_velocity(
            omega_left=-5.0, omega_right=5.0
        )

        # Opposite speeds → no translation
        assert np.isclose(v, 0.0, atol=1e-6)
        # Should rotate
        assert abs(omega) > 0

    def test_inverse_kinematics_straight(self):
        """Test: v>0, ω=0 → equal wheel speeds."""
        robot = Robot()
        omega_left, omega_right = robot.body_velocity_to_wheel_speeds(v=1.0, omega=0.0)

        assert np.isclose(omega_left, omega_right, atol=1e-6)

    def test_inverse_kinematics_rotation(self):
        """Test: v=0, ω>0 → opposite wheel speeds."""
        robot = Robot()
        omega_left, omega_right = robot.body_velocity_to_wheel_speeds(v=0.0, omega=1.0)

        assert np.isclose(omega_left, -omega_right, atol=1e-6)

    def test_roundtrip_conversion(self):
        """Test: Body→Wheel→Body should return original values."""
        robot = Robot()
        v_original = 1.5
        omega_original = 0.5

        # Convert to wheel speeds
        omega_l, omega_r = robot.body_velocity_to_wheel_speeds(
            v_original, omega_original
        )

        # Convert back to body velocities
        v_result, omega_result = robot.wheel_speeds_to_body_velocity(omega_l, omega_r)

        assert np.isclose(v_result, v_original, atol=1e-6)
        assert np.isclose(omega_result, omega_original, atol=1e-6)


class TestRobotReset:
    """Test robot reset functionality."""

    def test_reset_clears_velocities(self):
        """Test: Reset should zero out velocities."""
        robot = Robot()
        robot.update_kinematics(v_cmd=1.0, omega_cmd=1.0, dt=0.1)

        # Reset to new pose
        robot.reset(x=5.0, y=5.0, theta=np.pi)

        x, y, theta, v, omega = robot.get_state()
        assert x == 5.0
        assert y == 5.0
        assert np.isclose(theta, np.pi)
        assert v == 0.0
        assert omega == 0.0

    def test_reset_clears_trail(self):
        """Test: Reset should clear position trail."""
        robot = Robot()
        for _ in range(10):
            robot.update_kinematics(v_cmd=1.0, omega_cmd=0.0, dt=0.1)

        trail_before = len(robot.trail)
        assert trail_before > 1

        robot.reset(x=0.0, y=0.0, theta=0.0)
        assert len(robot.trail) == 1
