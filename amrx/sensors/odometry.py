"""
Odometry sensor model with realistic error.

Implements wheel encoder-based odometry as per Section 3 of specification:
- Systematic error (wheel diameter mismatch)
- Random error (slip, quantization)
- Output in robot frame for dead reckoning integration
"""

import numpy as np
from typing import Tuple, Optional


class OdometrySensor:
    """
    Wheel encoder-based odometry with realistic error model.

    Error sources (Section 3.1):
    - Wheel slip on floor
    - Unequal wheel diameters (manufacturing tolerance)
    - Encoder quantization
    - Kinematic model mismatch

    Output format (Section 3.3):
    Returns motion in robot frame: [Δx_robot, Δy_robot, Δθ]
    Student algorithms must integrate this to update pose estimate.
    """

    def __init__(
        self,
        systematic_error: float = 0.02,  # k_sys
        drift_per_meter: float = 0.05,  # k_r
        drift_per_radian: float = 0.05,  # k_θ
        random_state: Optional[np.random.RandomState] = None,
    ):
        """
        Initialize odometry sensor.

        Args:
            systematic_error: Systematic bias (e.g., 0.02 = 2% larger left wheel)
            drift_per_meter: Random drift coefficient per meter traveled (m)
            drift_per_radian: Random drift coefficient per radian rotated (rad)
            random_state: Random state for reproducibility
        """
        self.k_sys = systematic_error
        self.k_r = drift_per_meter
        self.k_theta = drift_per_radian

        # Random state for noise
        self.rng = random_state if random_state is not None else np.random.RandomState()

    def measure(
        self, delta_s_true: float, delta_theta_true: float
    ) -> Tuple[float, float, float]:
        """
        Measure odometry with realistic errors.

        Error model (Section 3.2):
        Δs_meas = Δs_true * (1 + k_sys) + N(0, k_r|Δs_true| + k_θ|Δθ_true|)
        Δθ_meas = Δθ_true + N(0, k_θ|Δθ_true| + k_r|Δs_true|)

        Output format (Section 3.3):
        Returns in robot frame coordinates:
        Δx_robot = Δs_meas * cos(Δθ_meas / 2)
        Δy_robot = Δs_meas * sin(Δθ_meas / 2)

        Special case (Section 3.4):
        If robot stationary (v < 0.001, |ω| < 0.001), return exact zeros.

        Args:
            delta_s_true: True linear displacement (meters)
            delta_theta_true: True angular displacement (radians)

        Returns:
            (delta_x_robot, delta_y_robot, delta_theta) in robot frame
        """
        # Special case: robot stationary
        if abs(delta_s_true) < 0.001 and abs(delta_theta_true) < 0.001:
            return 0.0, 0.0, 0.0

        # Calculate noise standard deviations
        sigma_s = self.k_r * abs(delta_s_true) + self.k_theta * abs(delta_theta_true)
        sigma_theta = self.k_theta * abs(delta_theta_true) + self.k_r * abs(delta_s_true)

        # Add systematic error to linear motion
        delta_s_sys = delta_s_true * (1.0 + self.k_sys)

        # Add random errors
        noise_s = self.rng.normal(0.0, sigma_s)
        noise_theta = self.rng.normal(0.0, sigma_theta)

        delta_s_meas = delta_s_sys + noise_s
        delta_theta_meas = delta_theta_true + noise_theta

        # Convert to robot frame (Section 3.3)
        # Use half-angle for better approximation of curved motion
        half_angle = delta_theta_meas / 2.0
        delta_x_robot = delta_s_meas * np.cos(half_angle)
        delta_y_robot = delta_s_meas * np.sin(half_angle)

        return delta_x_robot, delta_y_robot, delta_theta_meas

    def integrate_odometry(
        self,
        x: float,
        y: float,
        theta: float,
        delta_x_robot: float,
        delta_y_robot: float,
        delta_theta: float,
    ) -> Tuple[float, float, float]:
        """
        Integrate odometry measurement into global pose estimate.

        This is the dead reckoning update that students would implement.
        Provided here for testing and reference.

        Equations (from Section 3.3):
        x_{k+1} = x_k + Δx_robot * cos(θ_k) - Δy_robot * sin(θ_k)
        y_{k+1} = y_k + Δx_robot * sin(θ_k) + Δy_robot * cos(θ_k)
        θ_{k+1} = θ_k + Δθ

        Args:
            x: Current X estimate (meters)
            y: Current Y estimate (meters)
            theta: Current heading estimate (radians)
            delta_x_robot: Forward displacement in robot frame (meters)
            delta_y_robot: Lateral displacement in robot frame (meters)
            delta_theta: Angular displacement (radians)

        Returns:
            (x_new, y_new, theta_new) updated pose estimate
        """
        # Rotate robot-frame displacement to global frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_new = x + delta_x_robot * cos_theta - delta_y_robot * sin_theta
        y_new = y + delta_x_robot * sin_theta + delta_y_robot * cos_theta
        theta_new = theta + delta_theta

        # Normalize angle
        from amrx.utils.geometry import normalize_angle
        theta_new = normalize_angle(theta_new)

        return x_new, y_new, theta_new

    def __repr__(self) -> str:
        return (
            f"OdometrySensor(k_sys={self.k_sys:.3f}, "
            f"k_r={self.k_r:.3f}, k_θ={self.k_theta:.3f})"
        )
