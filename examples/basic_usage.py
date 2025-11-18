"""
Basic usage example for AMRx Simulator.

Demonstrates:
- Loading a world from JSON
- Creating a simulation
- Commanding the robot
- Reading sensor data
"""

from amrx import Simulation, World

# Load world from JSON file
world = World.from_json("maps/empty_box.json")
print(f"Loaded world: {world.name}")
print(f"  Walls: {len(world.walls)}")
print(f"  Landmarks: {len(world.landmarks)}")

# Create simulation
sim = Simulation(world, dt=0.1, random_seed=42)
print(f"\nInitialized {sim}")

# Get initial pose
x, y, theta = sim.get_robot_pose()
print(f"Initial pose: x={x:.2f}m, y={y:.2f}m, θ={theta:.2f}rad")

# Run simulation for a few steps
print("\nRunning simulation...")
for i in range(10):
    # Command robot to move forward
    data = sim.step(v_cmd=0.5, omega_cmd=0.1)

    # Print sensor data every few steps
    if i % 3 == 0:
        print(f"\n--- Step {i + 1} (t={data['time']:.1f}s) ---")

        # Ground truth (for evaluation only!)
        x, y, theta, v, omega = data["ground_truth"]
        print(f"Ground truth: x={x:.2f}, y={y:.2f}, θ={theta:.2f}")

        # Odometry
        if data["odometry"]:
            dx, dy, dtheta = data["odometry"]
            print(f"Odometry: Δx={dx:.3f}, Δy={dy:.3f}, Δθ={dtheta:.3f}")

        # LiDAR
        if data["lidar"] is not None:
            ranges = data["lidar"]
            print(f"LiDAR: {len(ranges)} rays, "
                  f"min={ranges.min():.2f}m, max={ranges.max():.2f}m")

        # Landmarks
        if data["landmarks"]:
            print(f"Landmarks detected: {len(data['landmarks'])}")
            for r, bearing, sig in data["landmarks"][:3]:  # Show first 3
                print(f"  - Landmark {sig}: range={r:.2f}m, bearing={bearing:.2f}rad")

        # Collision
        if data["collision"]:
            print("  WARNING: Collision detected!")

print("\n" + "="*50)
print("Simulation complete!")
final_x, final_y, final_theta = sim.get_robot_pose()
print(f"Final pose: x={final_x:.2f}m, y={final_y:.2f}m, θ={final_theta:.2f}rad")
print(f"Total steps: {sim.step_count}, Total time: {sim.time:.1f}s")
