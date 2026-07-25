from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CircularObstacle:
    center: Tuple[float, float]
    radius: float

    def collides(self, point: Tuple[float, float], margin: float = 0.0) -> bool:
        px, py = point
        cx, cy = self.center
        dist = np.hypot(px - cx, py - cy)
        return dist <= self.radius + margin

    def blocks_segment(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        """True if the segment p0->p1 passes through this obstacle's disk."""
        x0, y0 = p0
        x1, y1 = p1
        cx, cy = self.center
        dx, dy = x1 - x0, y1 - y0
        fx, fy = x0 - cx, y0 - cy
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-12:
            return (fx * fx + fy * fy) <= self.radius * self.radius
        t = max(0.0, min(1.0, -(fx * dx + fy * dy) / seg_len_sq))
        nx, ny = x0 + t * dx - cx, y0 + t * dy - cy
        return (nx * nx + ny * ny) <= self.radius * self.radius


@dataclass
class RectangularObstacle:
    x: float
    y: float
    width: float
    height: float

    def collides(self, point: Tuple[float, float], margin: float = 0.0) -> bool:
        px, py = point
        return (
            self.x - margin <= px <= self.x + self.width + margin
            and self.y - margin <= py <= self.y + self.height + margin
        )

    def blocks_segment(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        """True if the segment p0->p1 passes through this obstacle's rect (Liang-Barsky clip)."""
        x0, y0 = p0
        x1, y1 = p1
        x_min, x_max = self.x, self.x + self.width
        y_min, y_max = self.y, self.y + self.height
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in (
            (-dx, x0 - x_min),
            (dx, x_max - x0),
            (-dy, y0 - y_min),
            (dy, y_max - y0),
        ):
            if p == 0.0:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
        return t0 <= t1

