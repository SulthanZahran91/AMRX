"""
Unit tests for odometry sensor.

Tests from Section 11 (Testing & Validation Strategy).
"""

import pytest
import numpy as np

from amrx.sensors.odometry import OdometrySensor


class TestOdometryInitialization:
    """Test odometry sensor initialization."""

    def test_default_initialization(self):
        """Test: Default odometry parameters match spec."""
        odom = OdometrySensor()

        assert odom.k_sys == 0.02  # 2% systematic error
        assert odom.k_r == 0.05  # 5cm drift per meter
        assert odom.k_theta == 0.05  # 0.05 rad drift per radian

    def test_custom_initialization(self):
        """Test: Custom odometry parameters."""
        odom = OdometrySensor(
            systematic_error=0.01, drift_per_meter=0.1, drift_per_radian=0.1
        )

        assert odom.k_sys == 0.01
        assert odom.k_r == 0.1
        assert odom.k_theta == 0.1


class TestOdometryMeasurement:
    """Test odometry measurement with errors."""

    def test_stationary_returns_zeros(self):
        """Test from spec 3.4: Stationary robot returns exact zeros."""
        odom = OdometrySensor()

        dx, dy, dtheta = odom.measure(delta_s_true=0.0, delta_theta_true=0.0)

        assert dx == 0.0
        assert dy == 0.0
        assert dtheta == 0.0

    def test_small_motion_returns_zeros(self):
        """Test: Very small motion (below threshold) returns zeros."""
        odom = OdometrySensor()

        dx, dy, dtheta = odom.measure(delta_s_true=0.0001, delta_theta_true=0.0001)

        assert dx == 0.0
        assert dy == 0.0
        assert dtheta == 0.0

    def test_systematic_error_applied(self):
        """Test: Systematic error consistently biases measurements."""
        odom = OdometrySensor(
            systematic_error=0.1,  # 10% bias
            drift_per_meter=0.0,  # No random noise
            drift_per_radian=0.0,
        )

        # Pure forward motion, no rotation
        dx, dy, dtheta = odom.measure(delta_s_true=1.0, delta_theta_true=0.0)

        # With 10% systematic error, should measure ~1.1m
        assert abs(dx - 1.1) < 0.01
        assert abs(dy) < 0.01  # Should be near zero
        assert abs(dtheta) < 0.01  # Should be near zero

    def test_straight_motion_output_format(self):
        """Test: Straight motion produces correct robot-frame output."""
        odom = OdometrySensor(
            systematic_error=0.0, drift_per_meter=0.0, drift_per_radian=0.0
        )

        # 1 meter forward, no rotation
        dx, dy, dtheta = odom.measure(delta_s_true=1.0, delta_theta_true=0.0)

        # Should be approximately (1, 0, 0) in robot frame
        assert abs(dx - 1.0) < 0.01
        assert abs(dy) < 0.01
        assert abs(dtheta) < 0.01

    def test_pure_rotation_output_format(self):
        """Test: Pure rotation produces correct output."""
        odom = OdometrySensor(
            systematic_error=0.0, drift_per_meter=0.0, drift_per_radian=0.0
        )

        # No translation, 90 degree rotation
        dx, dy, dtheta = odom.measure(delta_s_true=0.0, delta_theta_true=np.pi / 2)

        # Should be approximately (0, 0, π/2)
        assert abs(dx) < 0.01
        assert abs(dy) < 0.01
        assert abs(dtheta - np.pi / 2) < 0.01

    def test_curved_motion_uses_half_angle(self):
        """Test: Curved motion uses half-angle approximation."""
        odom = OdometrySensor(
            systematic_error=0.0, drift_per_meter=0.0, drift_per_radian=0.0
        )

        # 1 meter forward with 90 degree turn
        # Half angle = 45 degrees
        dx, dy, dtheta = odom.measure(delta_s_true=1.0, delta_theta_true=np.pi / 2)

        # dx = 1.0 * cos(π/4) ≈ 0.707
        # dy = 1.0 * sin(π/4) ≈ 0.707
        assert abs(dx - 0.707) < 0.01
        assert abs(dy - 0.707) < 0.01
        assert abs(dtheta - np.pi / 2) < 0.01

    def test_noise_reproducibility(self):
        """Test: Same random seed produces identical measurements."""
        rng1 = np.random.RandomState(42)
        odom1 = OdometrySensor(random_state=rng1)
        dx1, dy1, dtheta1 = odom1.measure(1.0, 0.1)

        rng2 = np.random.RandomState(42)
        odom2 = OdometrySensor(random_state=rng2)
        dx2, dy2, dtheta2 = odom2.measure(1.0, 0.1)

        assert np.isclose(dx1, dx2)
        assert np.isclose(dy1, dy2)
        assert np.isclose(dtheta1, dtheta2)

    def test_noise_adds_variability(self):
        """Test: Random noise creates measurement variation."""
        odom = OdometrySensor()

        measurements = [odom.measure(1.0, 0.0) for _ in range(10)]
        dx_values = [m[0] for m in measurements]

        # Should have some variation
        assert np.std(dx_values) > 0.01


class TestOdometryIntegration:
    """Test odometry integration (dead reckoning)."""

    def test_integrate_straight_forward(self):
        """Test: Integrate straight forward motion."""
        odom = OdometrySensor()

        # Start at origin facing east
        x, y, theta = 0.0, 0.0, 0.0

        # Move 1m forward in robot frame
        x_new, y_new, theta_new = odom.integrate_odometry(
            x, y, theta, delta_x_robot=1.0, delta_y_robot=0.0, delta_theta=0.0
        )

        # Should move to (1, 0, 0)
        assert np.isclose(x_new, 1.0)
        assert np.isclose(y_new, 0.0)
        assert np.isclose(theta_new, 0.0)

    def test_integrate_with_rotation(self):
        """Test: Integrate motion with rotation."""
        odom = OdometrySensor()

        # Start at origin facing north
        x, y, theta = 0.0, 0.0, np.pi / 2

        # Move 1m forward in robot frame (which is north globally)
        x_new, y_new, theta_new = odom.integrate_odometry(
            x, y, theta, delta_x_robot=1.0, delta_y_robot=0.0, delta_theta=0.0
        )

        # Should move to (0, 1, π/2)
        assert np.isclose(x_new, 0.0, atol=1e-6)
        assert np.isclose(y_new, 1.0, atol=1e-6)
        assert np.isclose(theta_new, np.pi / 2)

    def test_integrate_lateral_motion(self):
        """Test: Integrate lateral (sideways) motion."""
        odom = OdometrySensor()

        # Start at origin facing east
        x, y, theta = 0.0, 0.0, 0.0

        # Move 1m left in robot frame
        x_new, y_new, theta_new = odom.integrate_odometry(
            x, y, theta, delta_x_robot=0.0, delta_y_robot=1.0, delta_theta=0.0
        )

        # Left in robot frame = +Y global when facing east
        assert np.isclose(x_new, 0.0, atol=1e-6)
        assert np.isclose(y_new, 1.0, atol=1e-6)
        assert np.isclose(theta_new, 0.0)

    def test_integrate_angle_normalization(self):
        """Test: Angle is normalized after integration."""
        odom = OdometrySensor()

        # Start facing east
        x, y, theta = 0.0, 0.0, 0.0

        # Rotate more than 2π
        x_new, y_new, theta_new = odom.integrate_odometry(
            x, y, theta, delta_x_robot=0.0, delta_y_robot=0.0, delta_theta=10 * np.pi
        )

        # Should normalize to [-π, π]
        assert -np.pi <= theta_new <= np.pi


class TestDeadReckoningDrift:
    """Test dead reckoning drift accumulation (integration test)."""

    def test_square_path_drift(self):
        """
        Test from spec 11.2: Drive robot in 1m square.
        With zero noise: should return to origin
        With noise: measure average endpoint error
        """
        # No noise case
        odom_perfect = OdometrySensor(
            systematic_error=0.0, drift_per_meter=0.0, drift_per_radian=0.0
        )

        x, y, theta = 0.0, 0.0, 0.0

        # Drive square: 4 sides of 1m with 90° turns
        for _ in range(4):
            # Forward 1m
            for _ in range(10):  # Break into 10cm segments
                dx, dy, dtheta = odom_perfect.measure(0.1, 0.0)
                x, y, theta = odom_perfect.integrate_odometry(x, y, theta, dx, dy, dtheta)

            # Turn 90 degrees
            dx, dy, dtheta = odom_perfect.measure(0.0, np.pi / 2)
            x, y, theta = odom_perfect.integrate_odometry(x, y, theta, dx, dy, dtheta)

        # Should return approximately to origin
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        # Angle should be back to 0 (or ±2π)
        assert abs(theta) < 0.01 or abs(abs(theta) - 2 * np.pi) < 0.01

    def test_long_distance_drift(self):
        """Test: Drift accumulates over long distances."""
        odom_noisy = OdometrySensor(
            systematic_error=0.02, drift_per_meter=0.05, drift_per_radian=0.05
        )

        x, y, theta = 0.0, 0.0, 0.0

        # Drive straight for 10 meters
        for _ in range(100):
            dx, dy, dtheta = odom_noisy.measure(0.1, 0.0)
            x, y, theta = odom_noisy.integrate_odometry(x, y, theta, dx, dy, dtheta)

        # Should have some error due to drift
        # Expected: ~10m forward, but with systematic error and noise
        error_from_expected = abs(x - 10.0)
        assert error_from_expected > 0.05  # Should have accumulated some error

    def test_forward_backward_cancellation(self):
        """Test: Forward then backward motion with errors."""
        odom = OdometrySensor(systematic_error=0.02)

        x, y, theta = 0.0, 0.0, 0.0

        # Forward 1m
        dx, dy, dtheta = odom.measure(1.0, 0.0)
        x, y, theta = odom.integrate_odometry(x, y, theta, dx, dy, dtheta)

        # Backward 1m (negative motion)
        dx, dy, dtheta = odom.measure(-1.0, 0.0)
        x, y, theta = odom.integrate_odometry(x, y, theta, dx, dy, dtheta)

        # Won't return exactly to origin due to systematic error
        # (systematic error is directional, so forward and backward compound)
        assert abs(x) > 0.005  # Should have residual error (relaxed tolerance)
