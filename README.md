# AMRx Simulator

A comprehensive simulator for differential drive robots with realistic sensor models and physics, designed for robotics education and algorithm development.

## Features

- **Differential Drive Kinematics**: Accurate simulation of two-wheeled robot motion with Euler integration
- **Realistic Sensors**:
  - 2D LiDAR (180° FOV, 361 rays, SICK LMS200 profile)
  - Odometry with realistic error models (systematic drift + random noise)
  - Landmark detector with field-of-view and occlusion handling
- **Robot Dynamics**: Motor constraints with acceleration limits and slew rate limiting
- **Collision Detection**: Circle-based robot collision with walls and cylindrical landmarks
- **Deterministic Simulation**: Reproducible results with random seed control
- **Comprehensive Test Suite**: 136 tests with 94% code coverage
- **Map Support**: Three pre-built test environments (Empty Box, Corridor, SLAM Arena)

## Installation

This project uses `uv` for package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```python
from amrx import Simulation, World

# Load a world map
world = World.from_json("maps/empty_box.json")

# Create simulation with deterministic random seed
sim = Simulation(world, dt=0.1, random_seed=42)

# Run simulation loop
for i in range(100):
    # Send velocity commands and get sensor data
    data = sim.step(v_cmd=0.5, omega_cmd=0.1)

    # Access sensor measurements
    odometry = data['odometry']           # (dx, dy, dtheta) in robot frame
    lidar_ranges = data['lidar']          # 361 range measurements
    landmarks = data['landmarks']         # List of (range, bearing, signature)
    ground_truth = data['ground_truth']   # (x, y, theta, v, omega)
    collision = data['collision']         # True if robot hit obstacle
    time = data['time']                   # Simulation time

    # Get current robot pose
    x, y, theta = sim.get_robot_pose()

    print(f"t={time:.2f}s: Robot at ({x:.2f}, {y:.2f}), detected {len(landmarks)} landmarks")
```

See [examples/](examples/) for complete demonstrations:
- `basic_usage.py` - Simple simulation loop with all sensors
- `dead_reckoning.py` - Odometry integration and drift visualization
- `landmark_tracking.py` - Landmark detection with occlusion

## Project Structure

```
amrx/
├── robot/          # Robot physics and kinematics
├── sensors/        # Sensor models (LiDAR, odometry, landmarks)
├── world/          # Environment representation
├── utils/          # Geometric algorithms
└── simulation.py   # Main simulation controller

maps/               # World definition files (JSON)
tests/              # Pytest test suite (136 tests, 94% coverage)
examples/           # Usage examples and demonstrations
```

## Coordinate System

- **Origin**: Bottom-left corner of arena
- **X-axis**: Points East (right)
- **Y-axis**: Points North (up)
- **Angular Zero**: θ = 0 points along +X axis
- **Angular Direction**: Counter-clockwise positive (right-hand rule)
- **Units**: Meters for distance, radians for angles

## Robot Parameters (Pioneer P3-DX Profile)

- Wheel base: 0.381 m
- Wheel radius: 0.0975 m
- Max linear velocity: 1.2 m/s
- Max angular velocity: 1.745 rad/s (100°/s)
- Max linear acceleration: 1.0 m/s²
- Max angular acceleration: 2.0 rad/s²
- Collision radius: 0.2 m

## Sensor Specifications

### LiDAR (SICK LMS200 Profile)
- Field of view: 180° (±90° from heading)
- Number of rays: 361
- Angular resolution: 0.5°
- Maximum range: 8.0 m
- Range noise: Gaussian, σ = 0.01 m

### Odometry
- Error model: Systematic + random components
- Systematic error: 2% of commanded velocity
- Random translation drift: σ = 0.05 m per meter traveled
- Random rotation drift: σ = 0.05 rad per radian rotated
- Output: (dx, dy, dtheta) in robot frame

### Landmark Detector
- Detection range: 5.0 m
- Field of view: 90° (±45° from heading)
- Range noise: σ = 0.05 m
- Bearing noise: σ = 0.02 rad (~1.1°)
- Occlusion: Full line-of-sight checking against walls
- Output: List of (range, bearing, signature) tuples

## API Reference

### Simulation Class

The main interface for running the simulator.

```python
from amrx import Simulation, World

# Initialization
sim = Simulation(
    world,                          # World object
    dt=0.1,                        # Time step (seconds)
    random_seed=None,              # Random seed for reproducibility
    enable_odometry=True,          # Enable odometry sensor
    enable_lidar=True,             # Enable LiDAR sensor
    enable_landmark_detector=True  # Enable landmark detector
)

# Step simulation (returns sensor data dictionary)
data = sim.step(v_cmd, omega_cmd)

# Reset simulation
sim.reset()                        # Reset to world start pose
sim.reset(x=1.0, y=2.0, theta=0)  # Reset to custom pose

# Run open-loop control
history = sim.run_open_loop(
    v_cmd=0.5,
    omega_cmd=0.1,
    duration=5.0,
    return_history=True
)

# Access robot state
x, y, theta = sim.get_robot_pose()
x, y, theta, v, omega = sim.robot.get_state()
```

### World Class

Represents the environment with walls and landmarks.

```python
from amrx import World

# Load from JSON
world = World.from_json("maps/empty_box.json")

# Create programmatically
world = World(name="Custom World")
world.add_wall(x1=0, y1=0, x2=10, y2=0)
world.add_landmark(x=5, y=5, signature=0)

# Check collisions
collision = world.check_robot_collision(
    robot_pos=(x, y),
    robot_radius=0.2
)
```

### Sensor Data Format

The `step()` method returns a dictionary with the following structure:

```python
{
    'odometry': (dx, dy, dtheta),           # Robot-frame motion increment
    'lidar': np.array([...]),               # 361 range measurements (meters)
    'landmarks': [(r, θ, sig), ...],        # List of detected landmarks
    'ground_truth': (x, y, theta, v, omega),# True robot state
    'collision': bool,                       # Collision flag
    'time': float                            # Simulation time (seconds)
}
```

### Map File Format

World maps are defined in JSON:

```json
{
    "name": "Empty Box",
    "bounds": {"width": 10.0, "height": 10.0},
    "robot_start": {"x": 5.0, "y": 5.0, "theta": 0.0},
    "walls": [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 0},
        {"x1": 10, "y1": 0, "x2": 10, "y2": 10}
    ],
    "landmarks": [
        {"x": 1, "y": 1, "signature": 0},
        {"x": 9, "y": 1, "signature": 1}
    ]
}
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=amrx --cov-report=html

# Format code
uv run black amrx/ tests/

# Type checking
uv run mypy amrx/
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for development guidelines and [TODO.md](TODO.md) for implementation progress.

## License

MIT License

## References

Based on the AMRx Simulator Technical Implementation Specification v1.0
