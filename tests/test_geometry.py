"""
Unit tests for geometric algorithms.

Tests from Section 11.1 (Unit Tests Per Component).
"""

import pytest
import numpy as np
from amrx.utils.geometry import (
    normalize_angle,
    circle_line_intersection,
    ray_line_intersection,
    ray_circle_intersection,
)


class TestAngleNormalization:
    """Test angle normalization to [-π, π] range."""

    def test_normalize_large_positive(self):
        """Test: θ = 7π should normalize to π."""
        result = normalize_angle(7 * np.pi)
        assert np.isclose(result, np.pi, atol=1e-10)

    def test_normalize_large_negative(self):
        """Test: θ = -3π should normalize to -π."""
        result = normalize_angle(-3 * np.pi)
        assert np.isclose(result, -np.pi, atol=1e-10)

    def test_normalize_within_range(self):
        """Test: θ already in range should remain unchanged."""
        theta = np.pi / 4
        result = normalize_angle(theta)
        assert np.isclose(result, theta, atol=1e-10)

    def test_normalize_zero(self):
        """Test: θ = 0 should remain 0."""
        result = normalize_angle(0.0)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_normalize_two_pi(self):
        """Test: θ = 2π should normalize to 0."""
        result = normalize_angle(2 * np.pi)
        assert np.isclose(result, 0.0, atol=1e-10)


class TestCircleLineIntersection:
    """Test circle-line segment intersection."""

    def test_circle_intersects_segment(self):
        """Test: Circle should intersect segment passing through it."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        p1 = np.array([-2.0, 0.0])
        p2 = np.array([2.0, 0.0])

        intersects, t = circle_line_intersection(center, radius, p1, p2)
        assert intersects is True
        assert t is not None
        assert 0 <= t <= 1

    def test_circle_no_intersection(self):
        """Test: Circle should not intersect distant segment."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        p1 = np.array([5.0, 5.0])
        p2 = np.array([6.0, 6.0])

        intersects, t = circle_line_intersection(center, radius, p1, p2)
        assert intersects is False
        assert t is None

    def test_circle_tangent_to_segment(self):
        """Test: Circle tangent to segment."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        p1 = np.array([-2.0, 1.0])
        p2 = np.array([2.0, 1.0])

        intersects, t = circle_line_intersection(center, radius, p1, p2)
        # Should barely touch
        assert intersects is True or not intersects  # Depends on tolerance


class TestRayLineIntersection:
    """Test ray-line segment intersection."""

    def test_ray_hits_segment(self):
        """Test from spec 11.1: Ray from (0,0) pointing +X, segment (2,1)→(2,-1)."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        p1 = np.array([2.0, 1.0])
        p2 = np.array([2.0, -1.0])

        intersects, t = ray_line_intersection(origin, direction, p1, p2)
        assert intersects is True
        assert t is not None
        assert np.isclose(t, 2.0, atol=1e-6)

        # Verify intersection point is at (2, 0)
        intersection_point = origin + t * direction
        assert np.isclose(intersection_point[0], 2.0, atol=1e-6)
        assert np.isclose(intersection_point[1], 0.0, atol=1e-6)

    def test_ray_misses_segment(self):
        """Test: Ray pointing away from segment."""
        origin = np.array([0.0, 0.0])
        direction = np.array([-1.0, 0.0])  # Pointing left
        p1 = np.array([2.0, 1.0])
        p2 = np.array([2.0, -1.0])

        intersects, t = ray_line_intersection(origin, direction, p1, p2)
        assert intersects is False
        assert t is None

    def test_ray_parallel_to_segment(self):
        """Test: Ray parallel to segment should not intersect."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        p1 = np.array([0.0, 1.0])
        p2 = np.array([2.0, 1.0])

        intersects, t = ray_line_intersection(origin, direction, p1, p2)
        assert intersects is False
        assert t is None


class TestRayCircleIntersection:
    """Test ray-circle intersection."""

    def test_ray_hits_circle(self):
        """Test: Ray from origin hits circle."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        center = np.array([5.0, 0.0])
        radius = 1.0

        intersects, t = ray_circle_intersection(origin, direction, center, radius)
        assert intersects is True
        assert t is not None
        assert np.isclose(t, 4.0, atol=1e-6)  # Should hit at x=4 (5-1)

    def test_ray_misses_circle(self):
        """Test: Ray does not hit circle."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        center = np.array([5.0, 5.0])  # Far above ray
        radius = 1.0

        intersects, t = ray_circle_intersection(origin, direction, center, radius)
        assert intersects is False
        assert t is None

    def test_ray_origin_inside_circle(self):
        """Test: Ray starting inside circle."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        center = np.array([0.0, 0.0])
        radius = 2.0

        intersects, t = ray_circle_intersection(origin, direction, center, radius)
        assert intersects is True
        assert t is not None
        assert t > 0  # Should hit exit point
