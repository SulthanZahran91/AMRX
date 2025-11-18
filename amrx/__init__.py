"""
AMRx Simulator: Technical Implementation for Autonomous Mobile Robotics

A comprehensive simulator for differential drive robots with realistic sensor models
and physics, designed for robotics education and algorithm development.
"""

__version__ = "1.0.0"
__author__ = "AMRx Team"

# Import what exists so far
from amrx.robot.robot import Robot
from amrx.world.world import World

__all__ = ["Robot", "World"]

# Will add later:
# from amrx.simulation import Simulation
