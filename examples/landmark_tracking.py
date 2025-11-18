"""
Landmark tracking example.

Demonstrates landmark detector capabilities:
- Detection range and field-of-view
- Occlusion by walls
- Measurement noise
- Converting measurements to global coordinates
"""

import numpy as np
from amrx import Simulation, World

# Load SLAM arena with multiple landmarks
world = World.from_json("maps/slam_arena.json")
sim = Simulation(world, dt=0.1, random_seed=42)

print("Landmark Tracking Example")
print("="*60)
print("Robot navigating through SLAM arena")
print("Demonstrating landmark detection with occlusion\n")

# Move robot through the environment
segments = [
    ("Initial position", 0.0, 0.0, 0.5),     # Pause to observe
    ("Move forward", 0.5, 0.0, 3.0),         # Drive forward
    ("Turn right", 0.0, -0.5, 3.14),         # 90 degree turn
    ("Move forward", 0.5, 0.0, 3.0),         # Drive forward
]

for segment_name, v_cmd, omega_cmd, duration in segments:
    print(f"\n{segment_name}: v={v_cmd:.1f} m/s, ω={omega_cmd:.2f} rad/s")
    print("-" * 60)

    num_steps = int(duration / sim.dt)
    for step in range(num_steps):
        data = sim.step(v_cmd, omega_cmd)

        # Show landmark detections periodically
        if step % 10 == 0:  # Every 1 second
            x, y, theta, v, omega = data["ground_truth"]
            landmarks = data["landmarks"]

            print(f"\nTime {data['time']:.1f}s - Robot at ({x:.2f}, {y:.2f}, θ={np.degrees(theta):.1f}°)")

            if landmarks:
                print(f"  Detected {len(landmarks)} landmark(s):")
                for i, (range_val, bearing, signature) in enumerate(landmarks):
                    # Convert to global coordinates
                    lm_x, lm_y = sim.landmark_detector.get_landmark_global_position(
                        x, y, theta, range_val, bearing
                    )

                    print(f"    [{i+1}] Signature {signature}: "
                          f"range={range_val:.2f}m, "
                          f"bearing={np.degrees(bearing):.1f}°, "
                          f"est_pos=({lm_x:.2f}, {lm_y:.2f})")
            else:
                print("  No landmarks detected (outside FOV or range)")

print("\n" + "="*60)
print("Landmark tracking complete!")
print("\nKey observations:")
print("- Landmarks are only detected within 5m range and 90° FOV")
print("- Detections are sorted by range (closest first)")
print("- Walls can occlude landmarks (line-of-sight required)")
print("- Measurements include noise (range ±5cm, bearing ±1.1°)")
print("- Each landmark has a unique signature for data association")
