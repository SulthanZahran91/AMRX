"""
AMRx Simulator: Technical Implementation for Autonomous Mobile Robotics

A comprehensive simulator for differential drive robots with realistic sensor models
and physics, designed for robotics education and algorithm development.
"""

__version__ = "1.0.0"
__author__ = "AMRx Team"

# Import only what exists so far
from amrx.robot.robot import Robot

__all__ = ["Robot"]

# Will add later:
# from amrx.world.world import World
# from amrx.simulation import Simulation
