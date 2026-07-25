from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List

import numpy as np


class Orientation(IntEnum):
    """Discrete orientations in multiples of 45 degrees."""

    RIGHT = 0
    UP_RIGHT = 1
    UP = 2
    UP_LEFT = 3
    LEFT = 4
    DOWN_LEFT = 5
    DOWN = 6
    DOWN_RIGHT = 7

    @staticmethod
    def from_angle(theta: float) -> "Orientation":
        """Quantize a continuous angle (rad) to the closest discrete orientation."""
        k = int(np.round(theta / (np.pi / 4.0))) % 8
        return Orientation(k)

    def to_angle(self) -> float:
        """Convert orientation index back to angle in radians."""
        return float(self.value) * (np.pi / 4.0)


@dataclass(frozen=True)
class MotionPrimitive:
    """
    A one-step action. Move: turn to `new_orientation`, step forward. Inject: no
    motion, only a world side-effect. Retract: step back to the previous pose in
    the traversal history (real motion, not a command undo).
    """

    new_orientation: Orientation
    is_inject: bool = False
    is_retract: bool = False


def allowed_primitives(current_orientation: Orientation) -> List[MotionPrimitive]:
    """The three forward moves from here: turn -45deg, 0, or +45deg, then step."""
    k = int(current_orientation)
    opts = [((k - 1) % 8), k % 8, ((k + 1) % 8)]
    return [MotionPrimitive(new_orientation=Orientation(o), is_inject=False) for o in opts]


def inject_primitive(orientation: Orientation) -> MotionPrimitive:
    """Create an inject primitive (robot does not move)."""
    return MotionPrimitive(new_orientation=orientation, is_inject=True)


def retract_primitive(orientation: Orientation) -> MotionPrimitive:
    """Retract primitive. `new_orientation` is unused (the pose comes from history)
    but kept for interface consistency."""
    return MotionPrimitive(new_orientation=orientation, is_inject=False, is_retract=True)


def apply_primitive(
    x: float,
    y: float,
    orientation: Orientation,
    primitive: MotionPrimitive,
    step_size: float = 1.0,
    *,
    pose_history: List[tuple[float, float, float]] | None = None,
):
    """Apply a primitive, returning the new (x, y, theta)."""
    if primitive.is_inject:
        # Inject is a side-effect only; robot pose does not change.
        return float(x), float(y), float(orientation.to_angle())

    if primitive.is_retract:
        if not pose_history or len(pose_history) < 2:
            # No-op when there is no previous step to retract to.
            return float(x), float(y), float(orientation.to_angle())
        # Retract to the previously visited pose (1 step back).
        x_prev, y_prev, theta_prev = pose_history[-2]
        return float(x_prev), float(y_prev), float(theta_prev)

    new_orientation = primitive.new_orientation
    theta = new_orientation.to_angle()
    # Grid-style motion: axial moves advance by `step_size` in one axis,
    # diagonal moves advance by `step_size` in both axes (not normalized).
    k = int(new_orientation) % 8
    if k == int(Orientation.RIGHT):
        dx, dy = 1.0, 0.0
    elif k == int(Orientation.UP_RIGHT):
        dx, dy = 1.0, 1.0
    elif k == int(Orientation.UP):
        dx, dy = 0.0, 1.0
    elif k == int(Orientation.UP_LEFT):
        dx, dy = -1.0, 1.0
    elif k == int(Orientation.LEFT):
        dx, dy = -1.0, 0.0
    elif k == int(Orientation.DOWN_LEFT):
        dx, dy = -1.0, -1.0
    elif k == int(Orientation.DOWN):
        dx, dy = 0.0, -1.0
    else:  # DOWN_RIGHT
        dx, dy = 1.0, -1.0

    x_new = x + step_size * dx
    y_new = y + step_size * dy
    return float(x_new), float(y_new), float(theta)

