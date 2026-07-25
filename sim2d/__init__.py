"""
Simple 2D robot simulation environment for motion planning.

Main entrypoint is `World2D` in `world.py`.
"""

from .viewport import DEFAULT_VIEW_X_LIMITS, DEFAULT_VIEW_Y_LIMITS
from .world import World2D
from .robot import DifferentialDriveRobot
from .obstacles import CircularObstacle, RectangularObstacle
from .primitives import (
    Orientation,
    MotionPrimitive,
    allowed_primitives,
    apply_primitive,
    inject_primitive,
    retract_primitive,
)

__all__ = [
    "DEFAULT_VIEW_X_LIMITS",
    "DEFAULT_VIEW_Y_LIMITS",
    "World2D",
    "DifferentialDriveRobot",
    "CircularObstacle",
    "RectangularObstacle",
    "Orientation",
    "MotionPrimitive",
    "allowed_primitives",
    "apply_primitive",
    "inject_primitive",
    "retract_primitive",
]

