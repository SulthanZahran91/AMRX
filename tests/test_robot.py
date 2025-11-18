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
        # Note: With dynamics, there's ramp-up time, so rotation is less than 1 radian
        # Rough check: position should have moved
        assert abs(x) > 0.1
        assert abs(y) > 0.1
        # With acceleration limits, won't reach full 1.0 radian
        # (ramp-up reduces average omega)
        assert 0.5 < theta < 1.0  # Some rotation, but less than ideal

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


class TestRobotDynamics:
    """Test robot dynamics with acceleration limits (Phase 6)."""

    def test_velocity_limited_to_max(self):
        """Test: Commanded velocity is clipped to maximum."""
        robot = Robot(max_linear_velocity=1.0)

        # Command velocity above maximum
        robot.update_kinematics(v_cmd=5.0, omega_cmd=0.0, dt=0.1)

        # Should be clipped to max
        _, _, _, v, _ = robot.get_state()
        assert v <= 1.0

    def test_angular_velocity_limited_to_max(self):
        """Test: Commanded angular velocity is clipped to maximum."""
        robot = Robot(max_angular_velocity=1.0)

        # Command angular velocity above maximum
        robot.update_kinematics(v_cmd=0.0, omega_cmd=10.0, dt=0.1)

        # Should be clipped to max
        _, _, _, _, omega = robot.get_state()
        assert omega <= 1.0

    def test_slew_rate_prevents_instantaneous_change(self):
        """Test from spec 2.4: Slew rate limiting prevents instant velocity changes."""
        robot = Robot(max_linear_accel=1.0)

        # Robot starts at rest, command high velocity
        robot.update_kinematics(v_cmd=10.0, omega_cmd=0.0, dt=0.1)

        # After 0.1s with a_max=1.0, max velocity change is 0.1 m/s
        _, _, _, v, _ = robot.get_state()
        assert v <= 0.1 + 0.01  # 0.1 + small tolerance

    def test_acceleration_ramp_up(self):
        """Test: Velocity ramps up gradually with acceleration limit."""
        robot = Robot(max_linear_accel=1.0)

        # Apply constant command for 1 second
        for _ in range(10):
            robot.update_kinematics(v_cmd=2.0, omega_cmd=0.0, dt=0.1)

        # After 1s, should reach approximately 1.0 m/s (a_max * t)
        _, _, _, v, _ = robot.get_state()
        assert 0.9 < v < 1.2  # Close to 1.0 m/s

    def test_angular_acceleration_ramp_up(self):
        """Test: Angular velocity ramps up gradually."""
        robot = Robot(max_angular_accel=2.0)

        # Apply constant command for 0.5 seconds
        for _ in range(5):
            robot.update_kinematics(v_cmd=0.0, omega_cmd=5.0, dt=0.1)

        # After 0.5s, should reach approximately 1.0 rad/s (alpha_max * t)
        _, _, _, _, omega = robot.get_state()
        assert 0.9 < omega < 1.2

    def test_deceleration_limited(self):
        """Test: Deceleration is also limited by acceleration constraint."""
        robot = Robot(max_linear_accel=1.0)

        # First accelerate to some velocity
        for _ in range(10):
            robot.update_kinematics(v_cmd=1.0, omega_cmd=0.0, dt=0.1)

        # Now command stop
        robot.update_kinematics(v_cmd=0.0, omega_cmd=0.0, dt=0.1)

        # Should not stop immediately, decel limited to a_max*dt
        _, _, _, v, _ = robot.get_state()
        assert v > 0.8  # Should still be moving

    def test_step_response(self):
        """Test from spec 2.4: Test step response to velocity command."""
        robot = Robot(max_linear_accel=1.0)

        velocities = []
        for _ in range(20):
            robot.update_kinematics(v_cmd=1.0, omega_cmd=0.0, dt=0.1)
            _, _, _, v, _ = robot.get_state()
            velocities.append(v)

        # Velocity should increase monotonically until reaching setpoint
        for i in range(len(velocities) - 1):
            # Either increasing or at steady state
            assert velocities[i + 1] >= velocities[i] - 0.01

        # Should eventually reach commanded velocity
        assert velocities[-1] >= 0.95

    def test_backwards_motion_dynamics(self):
        """Test: Dynamics work for backwards motion."""
        robot = Robot(max_linear_accel=1.0)

        # Command backwards motion
        for _ in range(5):
            robot.update_kinematics(v_cmd=-1.0, omega_cmd=0.0, dt=0.1)

        _, _, _, v, _ = robot.get_state()
        # Should be moving backwards but limited by acceleration
        assert v < 0
        assert v >= -0.6  # Limited by accel over 0.5s

    def test_combined_linear_and_angular_dynamics(self):
        """Test: Both linear and angular dynamics work simultaneously."""
        robot = Robot(max_linear_accel=1.0, max_angular_accel=2.0)

        # Command both linear and angular motion
        for _ in range(5):
            robot.update_kinematics(v_cmd=1.0, omega_cmd=2.0, dt=0.1)

        _, _, _, v, omega = robot.get_state()

        # Both should be ramping up
        assert 0.4 < v < 0.6  # ~0.5 m/s after 0.5s
        assert 0.9 < omega < 1.2  # ~1.0 rad/s after 0.5s

    def test_dynamics_with_zero_acceleration_limit(self):
        """Test: Zero acceleration limit prevents any motion."""
        robot = Robot(max_linear_accel=0.0)

        # Try to move
        robot.update_kinematics(v_cmd=1.0, omega_cmd=0.0, dt=0.1)

        _, _, _, v, _ = robot.get_state()
        # Should remain at zero
        assert v == 0.0
