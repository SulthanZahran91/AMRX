# AMRx Simulator

A comprehensive simulator for differential drive robots with realistic sensor models and physics, designed for robotics education and algorithm development.

## Features

- **Differential Drive Kinematics**: Accurate simulation of two-wheeled robot motion
- **Realistic Sensors**:
  - 2D LiDAR (180° FOV, 361 rays, SICK LMS200 profile)
  - Odometry with realistic error models (systematic drift + random noise)
  - Landmark detector with occlusion handling
- **Robot Dynamics**: Motor constraints with slew rate limiting
- **Collision Detection**: Circle-based robot with wall and obstacle interactions
- **Data Logging**: HDF5 format for efficient storage and analysis
- **Real-time Visualization**: PyQt6-based interactive interface
- **Deterministic Simulation**: Reproducible results with random seed control

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

# Create simulation
sim = Simulation(world, dt=0.1, headless=False, random_seed=42)

# Run simulation
for i in range(1000):
    # Get sensor data
    sensors = sim.step(v_cmd=1.0, omega_cmd=0.0)

    # Process odometry, lidar, landmarks
    odom = sensors['odometry']
    lidar = sensors['lidar']
    landmarks = sensors['landmarks']

sim.close()
```

## Project Structure

```
amrx/
├── robot/          # Robot physics and kinematics
├── sensors/        # Sensor models (LiDAR, odometry, landmarks)
├── world/          # Environment representation
├── visualization/  # PyQt6 real-time rendering
└── utils/          # Geometric algorithms

maps/               # World definition files (JSON)
tests/              # Pytest test suite
examples/           # Usage examples
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
- Collision radius: 0.2 m

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
