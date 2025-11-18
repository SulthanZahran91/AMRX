"""
Geometry data structures for world representation.

Defines walls (line segments) and landmarks (cylinders) as per Section 6.1.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class LineSegment:
    """
    Wall segment representation.

    A wall is defined by two endpoints in the global frame.
    Used for collision detection and LiDAR ray tracing.
    """

    start: np.ndarray  # (x, y) start point
    end: np.ndarray  # (x, y) end point

    def __init__(self, start: Tuple[float, float], end: Tuple[float, float]):
        """
        Create a line segment.

        Args:
            start: (x, y) coordinates of start point
            end: (x, y) coordinates of end point
        """
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)

    def length(self) -> float:
        """Get the length of the segment."""
        return float(np.linalg.norm(self.end - self.start))

    def direction(self) -> np.ndarray:
        """Get the normalized direction vector."""
        vec = self.end - self.start
        length = np.linalg.norm(vec)
        if length < 1e-10:
            return np.array([1.0, 0.0])
        return vec / length


@dataclass
class Landmark:
    """
    Landmark cylinder representation.

    Landmarks are cylindrical obstacles with unique signatures
    for data association in SLAM algorithms.
    """

    center: np.ndarray  # (x, y) center position
    radius: float  # meters (typically 0.1m)
    signature: int  # unique ID for data association

    def __init__(
        self, center: Tuple[float, float], radius: float = 0.1, signature: int = 0
    ):
        """
        Create a landmark.

        Args:
            center: (x, y) coordinates of center
            radius: Cylinder radius (meters)
            signature: Unique identifier for data association
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.signature = int(signature)

    def distance_to(self, point: np.ndarray) -> float:
        """
        Get distance from landmark center to a point.

        Args:
            point: (x, y) coordinates

        Returns:
            Euclidean distance (meters)
        """
        return float(np.linalg.norm(self.center - point))
