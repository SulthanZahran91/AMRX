"""
World environment representation.

Manages walls, landmarks, and provides collision detection.
Supports loading from JSON map files (Section 6.3).
"""

import json
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from amrx.world.geometry import LineSegment, Landmark
from amrx.utils.geometry import circle_line_intersection


class World:
    """
    World environment with walls and landmarks.

    Provides:
    - Map loading from JSON files
    - Collision detection for robot
    - Geometric queries for sensors
    """

    def __init__(self, name: str = "World"):
        """
        Create an empty world.

        Args:
            name: World name/description
        """
        self.name = name
        self.walls: List[LineSegment] = []
        self.landmarks: List[Landmark] = []
        self.robot_start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Bounding box (calculated from walls)
        self.x_min = 0.0
        self.x_max = 10.0
        self.y_min = 0.0
        self.y_max = 10.0

    @classmethod
    def from_json(cls, filepath: str) -> "World":
        """
        Load world from JSON map file.

        JSON format (Section 6.3):
        {
          "name": "World Name",
          "description": "...",
          "walls": [
            {"start": [x1, y1], "end": [x2, y2]},
            ...
          ],
          "landmarks": [
            {"pos": [x, y], "radius": 0.1, "sig": 0},
            ...
          ],
          "robot_start": {"x": 5.0, "y": 5.0, "theta": 0.0}
        }

        Args:
            filepath: Path to JSON map file

        Returns:
            World instance loaded from file
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {filepath}")

        with open(path, "r") as f:
            data = json.load(f)

        world = cls(name=data.get("name", "World"))

        # Load walls
        for wall_data in data.get("walls", []):
            wall = LineSegment(
                start=tuple(wall_data["start"]), end=tuple(wall_data["end"])
            )
            world.walls.append(wall)

        # Load landmarks
        for lm_data in data.get("landmarks", []):
            landmark = Landmark(
                center=tuple(lm_data["pos"]),
                radius=lm_data.get("radius", 0.1),
                signature=lm_data["sig"],
            )
            world.landmarks.append(landmark)

        # Load robot start pose
        if "robot_start" in data:
            start = data["robot_start"]
            world.robot_start_pose = (start["x"], start["y"], start["theta"])

        # Calculate bounding box
        world._calculate_bounds()

        return world

    def add_wall(self, start: Tuple[float, float], end: Tuple[float, float]) -> None:
        """
        Add a wall segment to the world.

        Args:
            start: (x, y) start point
            end: (x, y) end point
        """
        self.walls.append(LineSegment(start, end))
        self._calculate_bounds()

    def add_landmark(
        self, center: Tuple[float, float], radius: float = 0.1, signature: int = 0
    ) -> None:
        """
        Add a landmark to the world.

        Args:
            center: (x, y) center position
            radius: Cylinder radius (meters)
            signature: Unique identifier
        """
        self.landmarks.append(Landmark(center, radius, signature))

    def _calculate_bounds(self) -> None:
        """Calculate world bounding box from walls."""
        if not self.walls:
            return

        all_x = []
        all_y = []

        for wall in self.walls:
            all_x.extend([wall.start[0], wall.end[0]])
            all_y.extend([wall.start[1], wall.end[1]])

        self.x_min = min(all_x)
        self.x_max = max(all_x)
        self.y_min = min(all_y)
        self.y_max = max(all_y)

    def check_robot_collision(
        self, robot_pos: np.ndarray, robot_radius: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if robot collides with walls or landmarks.

        Uses circle-line segment intersection (Section 7.2).
        Robot is modeled as a circle with given radius.

        Args:
            robot_pos: Robot center position (x, y)
            robot_radius: Robot bounding circle radius (meters)

        Returns:
            (collides, object_type) where:
            - collides: True if collision detected
            - object_type: "wall" or "landmark" if collision, None otherwise
        """
        # Check wall collisions
        for wall in self.walls:
            intersects, _ = circle_line_intersection(
                robot_pos, robot_radius, wall.start, wall.end
            )
            if intersects:
                return True, "wall"

        # Check landmark collisions (circle-circle)
        for landmark in self.landmarks:
            distance = np.linalg.norm(robot_pos - landmark.center)
            if distance < (robot_radius + landmark.radius):
                return True, "landmark"

        return False, None

    def check_collision_along_path(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        robot_radius: float,
        num_checks: int = 10,
    ) -> bool:
        """
        Check for collisions along a path.

        Samples points along the path and checks each.
        Used for more robust collision detection.

        Args:
            start_pos: Path start position (x, y)
            end_pos: Path end position (x, y)
            robot_radius: Robot bounding circle radius
            num_checks: Number of points to sample along path

        Returns:
            True if collision detected anywhere along path
        """
        for i in range(num_checks + 1):
            t = i / num_checks
            pos = start_pos + t * (end_pos - start_pos)
            collides, _ = self.check_robot_collision(pos, robot_radius)
            if collides:
                return True
        return False

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get world bounding box.

        Returns:
            (x_min, x_max, y_min, y_max)
        """
        return self.x_min, self.x_max, self.y_min, self.y_max

    def get_landmark_by_signature(self, signature: int) -> Optional[Landmark]:
        """
        Get landmark by its signature ID.

        Args:
            signature: Landmark signature to find

        Returns:
            Landmark if found, None otherwise
        """
        for landmark in self.landmarks:
            if landmark.signature == signature:
                return landmark
        return None
