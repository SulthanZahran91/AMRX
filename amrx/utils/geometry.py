"""
Geometric algorithms and utility functions.

Implements exact algorithms from Section 7 of the specification:
- Angle normalization
- Ray-line segment intersection
- Ray-circle intersection
- Circle-line segment intersection
"""

import numpy as np
from typing import Tuple, Optional


def normalize_angle(theta: float) -> float:
    """
    Normalize angle to [-π, π] range.

    Uses atan2(sin(θ), cos(θ)) to ensure proper wrapping.
    This prevents unbounded accumulation and simplifies angle differencing.

    Args:
        theta: Angle in radians (can be any value)

    Returns:
        Normalized angle in range [-π, π]

    Example:
        >>> normalize_angle(7 * np.pi)
        3.141592653589793  # π
        >>> normalize_angle(-3 * np.pi)
        -3.141592653589793  # -π
    """
    return np.arctan2(np.sin(theta), np.cos(theta))


def circle_line_intersection(
    center: np.ndarray,
    radius: float,
    p1: np.ndarray,
    p2: np.ndarray
) -> Tuple[bool, Optional[float]]:
    """
    Check if a circle intersects with a line segment.

    Uses parametric line representation and quadratic formula.
    Returns the parameter t ∈ [0, 1] if intersection exists.

    Algorithm from Section 7.2:
    1. Vector from P1 to P2: d = P2 - P1
    2. Vector from P1 to C: f = P1 - C
    3. Quadratic coefficients:
       - a = d · d
       - b = 2(f · d)
       - c = f · f - r²
    4. Discriminant: Δ = b² - 4ac
    5. If Δ < 0: No intersection
    6. Otherwise: t = (-b ± √Δ) / 2a
    7. Valid if t ∈ [0, 1]

    Args:
        center: Circle center (x, y)
        radius: Circle radius
        p1: Line segment start point
        p2: Line segment end point

    Returns:
        (intersects, t) where:
        - intersects: True if circle and segment intersect
        - t: Parameter value in [0, 1] if intersection exists, None otherwise
    """
    d = p2 - p1
    f = p1 - center

    a = np.dot(d, d)
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return False, None

    # Use negative root for nearest intersection
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Check if either intersection is on the segment
    if 0 <= t1 <= 1:
        return True, t1
    elif 0 <= t2 <= 1:
        return True, t2
    else:
        return False, None


def ray_line_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray
) -> Tuple[bool, Optional[float]]:
    """
    Find intersection of a ray with a line segment.

    Uses parametric forms:
    - Ray: R(t) = O + t*d, t ≥ 0
    - Segment: S(s) = P1 + s*(P2 - P1), s ∈ [0, 1]

    Solves R(t) = S(s) using 2D cross product.

    Algorithm from Section 7.3:
    t = (P1 - O) × (P2 - P1) / (d × (P2 - P1))

    Where × is 2D cross product: (ax, ay) × (bx, by) = ax*by - ay*bx

    Args:
        origin: Ray origin point (x, y)
        direction: Ray direction (unit vector preferred)
        p1: Line segment start point
        p2: Line segment end point

    Returns:
        (intersects, t) where:
        - intersects: True if ray hits segment
        - t: Distance along ray to intersection (t > 0), None otherwise
    """
    segment_vec = p2 - p1
    to_p1 = p1 - origin

    # 2D cross product
    def cross_2d(v1: np.ndarray, v2: np.ndarray) -> float:
        return v1[0] * v2[1] - v1[1] * v2[0]

    denom = cross_2d(direction, segment_vec)

    # Parallel rays
    if abs(denom) < 1e-10:
        return False, None

    t = cross_2d(to_p1, segment_vec) / denom
    s = cross_2d(to_p1, direction) / denom

    # Valid intersection: t > 0 (forward along ray) and s ∈ [0, 1] (on segment)
    if t > 1e-10 and 0 <= s <= 1:
        return True, t
    else:
        return False, None


def ray_circle_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float
) -> Tuple[bool, Optional[float]]:
    """
    Find intersection of a ray with a circle.

    Similar to circle-line intersection but t is unbounded (t > 0).

    Algorithm from Section 7.4:
    1. f = O - C
    2. a = 1 (assuming unit direction vector)
    3. b = 2(f · d)
    4. c = f · f - r²
    5. Δ = b² - 4c
    6. If Δ < 0: No hit
    7. t = (-b - √Δ) / 2 (use negative root for nearest intersection)
    8. Valid if t > 0

    Args:
        origin: Ray origin point (x, y)
        direction: Ray direction (should be unit vector)
        center: Circle center (x, y)
        radius: Circle radius

    Returns:
        (intersects, t) where:
        - intersects: True if ray hits circle
        - t: Distance along ray to intersection (t > 0), None otherwise
    """
    f = origin - center

    # Assuming direction is unit vector, a = 1
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(f, direction)
    c = np.dot(f, f) - radius * radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return False, None

    # Use negative root for nearest intersection
    sqrt_disc = np.sqrt(discriminant)
    t = (-b - sqrt_disc) / (2 * a)

    # Valid if t > 0 (forward along ray)
    if t > 1e-10:
        return True, t
    else:
        # Try positive root
        t = (-b + sqrt_disc) / (2 * a)
        if t > 1e-10:
            return True, t
        else:
            return False, None
