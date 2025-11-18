"""
Unit tests for Simulation class.

Tests integration of all components.
"""

import pytest
import numpy as np

from amrx import Simulation, World


class TestSimulationInitialization:
    """Test simulation initialization."""

    def test_basic_initialization(self):
        """Test: Basic simulation setup."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        assert sim.dt == 0.1
        assert sim.time == 0.0
        assert sim.step_count == 0
        assert sim.robot is not None
        assert sim.world is world

    def test_initialization_with_seed(self):
        """Test: Initialization with random seed."""
        world = World.from_json("maps/empty_box.json")
        sim1 = Simulation(world, random_seed=42)
        sim2 = Simulation(world, random_seed=42)

        # Should produce identical initial states
        assert sim1.get_robot_pose() == sim2.get_robot_pose()

    def test_robot_starts_at_world_position(self):
        """Test: Robot initializes at world's start position."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world)

        x, y, theta = sim.get_robot_pose()
        wx, wy, wtheta = world.robot_start_pose

        assert x == wx
        assert y == wy
        assert theta == wtheta

    def test_sensors_enabled_by_default(self):
        """Test: All sensors enabled by default."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world)

        assert sim.odometry is not None
        assert sim.lidar is not None
        assert sim.landmark_detector is not None

    def test_disable_sensors(self):
        """Test: Sensors can be disabled."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(
            world,
            enable_odometry=False,
            enable_lidar=False,
            enable_landmark_detector=False,
        )

        assert sim.odometry is None
        assert sim.lidar is None
        assert sim.landmark_detector is None


class TestSimulationStep:
    """Test simulation stepping."""

    def test_single_step(self):
        """Test: Single simulation step."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        data = sim.step(v_cmd=1.0, omega_cmd=0.0)

        assert "odometry" in data
        assert "lidar" in data
        assert "landmarks" in data
        assert "ground_truth" in data
        assert "collision" in data
        assert "time" in data

        # Time should advance
        assert data["time"] == 0.1

    def test_multiple_steps(self):
        """Test: Multiple simulation steps."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        for i in range(10):
            data = sim.step(v_cmd=0.5, omega_cmd=0.0)
            assert np.isclose(data["time"], (i + 1) * 0.1)

        assert sim.step_count == 10

    def test_robot_moves_forward(self):
        """Test: Robot moves in response to commands."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1, random_seed=42)

        x_initial, y_initial, theta_initial = sim.get_robot_pose()

        # Move forward for 1 second
        for _ in range(10):
            sim.step(v_cmd=0.5, omega_cmd=0.0)

        x_final, y_final, theta_final = sim.get_robot_pose()

        # Should have moved forward (in +X direction since theta=0)
        assert x_final > x_initial
        # Y and theta should be approximately unchanged
        assert np.isclose(y_final, y_initial, atol=0.1)
        assert np.isclose(theta_final, theta_initial, atol=0.1)

    def test_robot_rotates(self):
        """Test: Robot rotates in response to angular commands."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        theta_initial = sim.get_robot_pose()[2]

        # Rotate for 1 second
        for _ in range(10):
            sim.step(v_cmd=0.0, omega_cmd=1.0)

        theta_final = sim.get_robot_pose()[2]

        # Should have rotated (with dynamics, less than 1.0 rad)
        assert abs(theta_final - theta_initial) > 0.5


class TestCollisionDetection:
    """Test collision handling."""

    def test_collision_stops_robot(self):
        """Test: Collision stops robot motion."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        # Move robot very close to wall
        sim.robot.reset(x=0.25, y=5.0, theta=np.pi)  # Facing wall at x=0

        # Try to move into wall
        data = sim.step(v_cmd=1.0, omega_cmd=0.0)

        # Should detect collision
        if data["collision"]:
            # Robot should not have moved through wall
            x, y, theta = sim.get_robot_pose()
            assert x >= 0.2  # Collision radius

    def test_no_collision_in_open_space(self):
        """Test: No collision in open space."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        # Move in open space
        data = sim.step(v_cmd=0.5, omega_cmd=0.0)

        assert data["collision"] is False


class TestSensorData:
    """Test sensor data output."""

    def test_odometry_output_format(self):
        """Test: Odometry data has correct format."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        data = sim.step(v_cmd=0.5, omega_cmd=0.0)

        odom = data["odometry"]
        assert odom is not None
        assert len(odom) == 3  # (dx, dy, dtheta)

    def test_lidar_output_format(self):
        """Test: LiDAR data has correct format."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        data = sim.step(v_cmd=0.0, omega_cmd=0.0)

        lidar = data["lidar"]
        assert lidar is not None
        assert len(lidar) == 361  # Default LiDAR rays

    def test_landmark_output_format(self):
        """Test: Landmark data has correct format."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        data = sim.step(v_cmd=0.0, omega_cmd=0.0)

        landmarks = data["landmarks"]
        assert landmarks is not None
        assert isinstance(landmarks, list)

    def test_ground_truth_output(self):
        """Test: Ground truth included in output."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        data = sim.step(v_cmd=0.0, omega_cmd=0.0)

        gt = data["ground_truth"]
        assert len(gt) == 5  # (x, y, theta, v, omega)

    def test_disabled_sensors_return_none(self):
        """Test: Disabled sensors return None."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, enable_odometry=False)

        data = sim.step(v_cmd=0.0, omega_cmd=0.0)

        assert data["odometry"] is None


class TestSimulationReset:
    """Test simulation reset."""

    def test_reset_to_initial_state(self):
        """Test: Reset returns to initial state."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        initial_pose = sim.get_robot_pose()

        # Move robot
        for _ in range(10):
            sim.step(v_cmd=1.0, omega_cmd=0.5)

        # Verify robot moved
        moved_pose = sim.get_robot_pose()
        assert moved_pose != initial_pose

        # Reset
        sim.reset()

        # Should be back at initial pose
        reset_pose = sim.get_robot_pose()
        assert reset_pose == initial_pose

    def test_reset_time_and_step_count(self):
        """Test: Reset clears time and step count."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        # Run some steps
        for _ in range(10):
            sim.step(v_cmd=0.5, omega_cmd=0.0)

        assert sim.time > 0
        assert sim.step_count > 0

        # Reset
        sim.reset()

        assert sim.time == 0.0
        assert sim.step_count == 0

    def test_reset_to_custom_pose(self):
        """Test: Reset to custom pose."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        # Reset to custom position
        sim.reset(x=3.0, y=4.0, theta=np.pi / 2)

        x, y, theta = sim.get_robot_pose()
        assert x == 3.0
        assert y == 4.0
        assert np.isclose(theta, np.pi / 2)


class TestOpenLoopExecution:
    """Test open-loop control."""

    def test_run_open_loop(self):
        """Test: Run open-loop for duration."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        # Run for 1 second
        sim.run_open_loop(v_cmd=0.5, omega_cmd=0.0, duration=1.0)

        # Should have run 10 steps
        assert sim.step_count == 10
        assert np.isclose(sim.time, 1.0)

    def test_run_open_loop_with_history(self):
        """Test: Run open-loop and collect history."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        history = sim.run_open_loop(
            v_cmd=0.5, omega_cmd=0.0, duration=0.5, return_history=True
        )

        assert history is not None
        assert "time" in history
        assert "ground_truth" in history
        assert len(history["time"]) == 5  # 0.5s / 0.1s


class TestDeterministicBehavior:
    """Test deterministic simulation."""

    def test_same_seed_same_results(self):
        """Test: Same random seed produces identical results."""
        world = World.from_json("maps/empty_box.json")

        sim1 = Simulation(world, random_seed=42, dt=0.1)
        sim2 = Simulation(world, random_seed=42, dt=0.1)

        # Run both simulations
        for _ in range(10):
            data1 = sim1.step(v_cmd=1.0, omega_cmd=0.1)
            data2 = sim2.step(v_cmd=1.0, omega_cmd=0.1)

            # Ground truth should be identical
            assert data1["ground_truth"] == data2["ground_truth"]

            # Sensor data should be identical (same noise)
            if data1["odometry"] is not None and data2["odometry"] is not None:
                assert np.allclose(data1["odometry"], data2["odometry"])


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test: String representation."""
        world = World.from_json("maps/empty_box.json")
        sim = Simulation(world, dt=0.1)

        repr_str = repr(sim)
        assert "Simulation" in repr_str
        assert "Empty Box" in repr_str  # World name
        assert "0.1" in repr_str  # dt
