from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DifferentialDriveRobot:
    """Robot model for collision checking and drawing (currently just a radius)."""

    radius: float = 0.2

