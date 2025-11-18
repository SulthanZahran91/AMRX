"""
Dead reckoning example.

Demonstrates odometry-based localization and drift accumulation.
Compares estimated pose (from odometry integration) vs ground truth.
"""

import numpy as np
from amrx import Simulation, World

# Load world
world = World.from_json("maps/empty_box.json")
sim = Simulation(world, dt=0.1, random_seed=42)

# Initialize dead reckoning estimate
x_est, y_est, theta_est = sim.get_robot_pose()

print("Dead Reckoning Example")
print("="*60)
print("Integrating noisy odometry to track robot position")
print("Compare estimated pose vs ground truth to see drift\n")

# Drive in a square pattern
segments = [
    ("Forward", 0.5, 0.0, 2.0),      # Move forward 2m
    ("Turn left", 0.0, np.pi/4, 2.0), # Turn 90 degrees
    ("Forward", 0.5, 0.0, 2.0),      # Move forward 2m
    ("Turn left", 0.0, np.pi/4, 2.0), # Turn 90 degrees
    ("Forward", 0.5, 0.0, 2.0),      # Move forward 2m
    ("Turn left", 0.0, np.pi/4, 2.0), # Turn 90 degrees
    ("Forward", 0.5, 0.0, 2.0),      # Move forward 2m
    ("Turn left", 0.0, np.pi/4, 2.0), # Turn 90 degrees (back to start heading)
]

for segment_name, v_cmd, omega_cmd, duration in segments:
    print(f"\n{segment_name}: v={v_cmd:.1f} m/s, ω={omega_cmd:.2f} rad/s for {duration:.1f}s")

    num_steps = int(duration / sim.dt)
    for _ in range(num_steps):
        data = sim.step(v_cmd, omega_cmd)

        # Integrate odometry for dead reckoning
        if data["odometry"]:
            dx_robot, dy_robot, dtheta = data["odometry"]

            # Transform from robot frame to global frame
            cos_theta = np.cos(theta_est)
            sin_theta = np.sin(theta_est)

            x_est += dx_robot * cos_theta - dy_robot * sin_theta
            y_est += dx_robot * sin_theta + dy_robot * cos_theta
            theta_est += dtheta

            # Normalize angle
            from amrx.utils.geometry import normalize_angle
            theta_est = normalize_angle(theta_est)

    # Compare estimate vs ground truth
    x_true, y_true, theta_true = sim.get_robot_pose()

    error_x = x_true - x_est
    error_y = y_true - y_est
    error_pos = np.sqrt(error_x**2 + error_y**2)
    error_theta = normalize_angle(theta_true - theta_est)

    print(f"  Estimated: x={x_est:.3f}, y={y_est:.3f}, θ={theta_est:.3f}")
    print(f"  True:      x={x_true:.3f}, y={y_true:.3f}, θ={theta_true:.3f}")
    print(f"  Error:     Δpos={error_pos:.3f}m, Δθ={error_theta:.3f}rad")

print("\n" + "="*60)
print("Square pattern complete!")
print("\nFinal Comparison:")
x_true, y_true, theta_true = sim.get_robot_pose()
error_x = x_true - x_est
error_y = y_true - y_est
error_pos = np.sqrt(error_x**2 + error_y**2)
error_theta = normalize_angle(theta_true - theta_est)

print(f"Dead Reckoning:  x={x_est:.3f}m, y={y_est:.3f}m, θ={theta_est:.3f}rad")
print(f"Ground Truth:    x={x_true:.3f}m, y={y_true:.3f}m, θ={theta_true:.3f}rad")
print(f"Position Error:  {error_pos:.3f}m")
print(f"Heading Error:   {error_theta:.3f}rad ({np.degrees(error_theta):.1f}°)")
print("\nNote: This drift is why robots need Kalman filters or SLAM!")
