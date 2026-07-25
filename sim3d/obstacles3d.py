from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SphericalObstacle:
    """A solid ball obstacle in 3D."""

    center: Tuple[float, float, float]
    radius: float

    def collides(self, point: Tuple[float, float, float], margin: float = 0.0) -> bool:
        px, py, pz = point
        cx, cy, cz = self.center
        dist = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2))
        return dist <= self.radius + margin


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Combined rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll), all angles in
    radians. This is the single rotation convention shared by `BoxObstacle`
    and `World3D.compute_healthy_mask`'s voxel rasterization.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


@dataclass
class BoxObstacle:
    """
    Box obstacle (a "wall" in 3D), optionally rotated.

    Defined by the lower corner (x, y, z) plus extents (width, depth, height)
    along the x, y and z axes respectively (before rotation), and an
    optional (yaw, pitch, roll) rotation in radians applied about the box's
    own centroid via `rotation_matrix()`. Thin extents make a wall/slab.
    """

    x: float
    y: float
    z: float
    width: float   # extent along x
    depth: float   # extent along y
    height: float  # extent along z
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def centroid(self) -> Tuple[float, float, float]:
        return (
            self.x + self.width / 2.0,
            self.y + self.depth / 2.0,
            self.z + self.height / 2.0,
        )

    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix(self.yaw, self.pitch, self.roll)

    def collides(self, point: Tuple[float, float, float], margin: float = 0.0) -> bool:
        px, py, pz = point
        cx, cy, cz = self.centroid()
        d = np.array([px - cx, py - cy, pz - cz])
        # Inverse of an orthonormal rotation is its transpose.
        lx, ly, lz = self.rotation_matrix().T @ d
        hw, hd, hh = self.width / 2.0, self.depth / 2.0, self.height / 2.0
        return (
            -hw - margin <= lx <= hw + margin
            and -hd - margin <= ly <= hd + margin
            and -hh - margin <= lz <= hh + margin
        )

    def corners(self) -> np.ndarray:
        """Return the 8 (rotated) corner coordinates as an (8, 3) array (for rendering)."""
        hw, hd, hh = self.width / 2.0, self.depth / 2.0, self.height / 2.0
        local = np.array(
            [
                [-hw, -hd, -hh],
                [hw, -hd, -hh],
                [hw, hd, -hh],
                [-hw, hd, -hh],
                [-hw, -hd, hh],
                [hw, -hd, hh],
                [hw, hd, hh],
                [-hw, hd, hh],
            ]
        )
        rotated = local @ self.rotation_matrix().T
        return rotated + np.array(self.centroid())