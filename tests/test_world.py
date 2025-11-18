"""
Unit tests for World module.

Tests world loading, collision detection, and geometry.
"""

import pytest
import numpy as np
from pathlib import Path

from amrx.world.world import World
from amrx.world.geometry import LineSegment, Landmark


class TestLineSegment:
    """Test LineSegment data structure."""

    def test_creation(self):
        """Test: Create line segment."""
        seg = LineSegment(start=(0, 0), end=(1, 0))
        assert np.allclose(seg.start, [0, 0])
        assert np.allclose(seg.end, [1, 0])

    def test_length(self):
        """Test: Calculate segment length."""
        seg = LineSegment(start=(0, 0), end=(3, 4))
        assert np.isclose(seg.length(), 5.0)

    def test_direction(self):
        """Test: Get normalized direction vector."""
        seg = LineSegment(start=(0, 0), end=(1, 0))
        direction = seg.direction()
        assert np.allclose(direction, [1, 0])
        assert np.isclose(np.linalg.norm(direction), 1.0)


class TestLandmark:
    """Test Landmark data structure."""

    def test_creation(self):
        """Test: Create landmark."""
        lm = Landmark(center=(5, 5), radius=0.1, signature=42)
        assert np.allclose(lm.center, [5, 5])
        assert lm.radius == 0.1
        assert lm.signature == 42

    def test_distance_to(self):
        """Test: Calculate distance to point."""
        lm = Landmark(center=(0, 0), radius=0.1, signature=0)
        distance = lm.distance_to(np.array([3, 4]))
        assert np.isclose(distance, 5.0)


class TestWorldCreation:
    """Test World initialization and construction."""

    def test_empty_world(self):
        """Test: Create empty world."""
        world = World(name="Test")
        assert world.name == "Test"
        assert len(world.walls) == 0
        assert len(world.landmarks) == 0

    def test_add_wall(self):
        """Test: Add wall to world."""
        world = World()
        world.add_wall(start=(0, 0), end=(10, 0))
        assert len(world.walls) == 1

    def test_add_landmark(self):
        """Test: Add landmark to world."""
        world = World()
        world.add_landmark(center=(5, 5), radius=0.1, signature=0)
        assert len(world.landmarks) == 1

    def test_bounds_calculation(self):
        """Test: Bounding box calculated from walls."""
        world = World()
        world.add_wall(start=(0, 0), end=(10, 0))
        world.add_wall(start=(10, 0), end=(10, 8))
        world.add_wall(start=(10, 8), end=(0, 8))
        world.add_wall(start=(0, 8), end=(0, 0))

        x_min, x_max, y_min, y_max = world.get_bounds()
        assert x_min == 0
        assert x_max == 10
        assert y_min == 0
        assert y_max == 8


class TestWorldLoading:
    """Test loading worlds from JSON files."""

    def test_load_empty_box(self):
        """Test: Load empty box environment."""
        world = World.from_json("maps/empty_box.json")
        assert world.name == "Empty Box"
        assert len(world.walls) == 4  # Four walls
        assert len(world.landmarks) == 4  # Four corner landmarks

    def test_load_corridor(self):
        """Test: Load corridor environment."""
        world = World.from_json("maps/corridor.json")
        assert world.name == "Corridor"
        assert len(world.walls) == 4
        assert len(world.landmarks) == 6

    def test_load_slam_arena(self):
        """Test: Load SLAM arena environment."""
        world = World.from_json("maps/slam_arena.json")
        assert world.name == "SLAM Arena"
        assert len(world.walls) == 8  # Outer + inner walls
        assert len(world.landmarks) == 12

    def test_load_nonexistent_file(self):
        """Test: Loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            World.from_json("maps/does_not_exist.json")

    def test_robot_start_pose(self):
        """Test: Robot start pose loaded from JSON."""
        world = World.from_json("maps/empty_box.json")
        x, y, theta = world.robot_start_pose
        assert x == 5.0
        assert y == 5.0
        assert theta == 0.0


class TestCollisionDetection:
    """Test collision detection."""

    def test_no_collision_in_open_space(self):
        """Test: No collision in center of empty box."""
        world = World.from_json("maps/empty_box.json")
        robot_pos = np.array([5.0, 5.0])
        robot_radius = 0.2

        collides, _ = world.check_robot_collision(robot_pos, robot_radius)
        assert collides is False

    def test_collision_with_wall(self):
        """Test: Collision detected when robot touches wall."""
        world = World.from_json("maps/empty_box.json")
        # Position very close to bottom wall (y=0)
        robot_pos = np.array([5.0, 0.1])
        robot_radius = 0.2

        collides, obj_type = world.check_robot_collision(robot_pos, robot_radius)
        assert collides is True
        assert obj_type == "wall"

    def test_collision_with_landmark(self):
        """Test: Collision detected when robot hits landmark."""
        world = World.from_json("maps/empty_box.json")
        # Position near landmark at (1, 1)
        robot_pos = np.array([1.0, 1.0])
        robot_radius = 0.2

        collides, obj_type = world.check_robot_collision(robot_pos, robot_radius)
        assert collides is True
        assert obj_type == "landmark"

    def test_collision_along_path(self):
        """Test: Detect collision along movement path."""
        world = World.from_json("maps/empty_box.json")
        # Path that crosses bottom wall
        start_pos = np.array([5.0, 1.0])
        end_pos = np.array([5.0, -1.0])
        robot_radius = 0.2

        collides = world.check_collision_along_path(
            start_pos, end_pos, robot_radius, num_checks=10
        )
        assert collides is True

    def test_no_collision_along_safe_path(self):
        """Test: No collision along safe path."""
        world = World.from_json("maps/empty_box.json")
        # Path in open space
        start_pos = np.array([3.0, 5.0])
        end_pos = np.array([7.0, 5.0])
        robot_radius = 0.2

        collides = world.check_collision_along_path(
            start_pos, end_pos, robot_radius, num_checks=10
        )
        assert collides is False


class TestLandmarkQueries:
    """Test landmark query functions."""

    def test_get_landmark_by_signature(self):
        """Test: Find landmark by signature."""
        world = World.from_json("maps/empty_box.json")
        landmark = world.get_landmark_by_signature(0)

        assert landmark is not None
        assert landmark.signature == 0
        assert np.allclose(landmark.center, [1, 1])

    def test_get_nonexistent_landmark(self):
        """Test: Return None for nonexistent signature."""
        world = World.from_json("maps/empty_box.json")
        landmark = world.get_landmark_by_signature(999)
        assert landmark is None
