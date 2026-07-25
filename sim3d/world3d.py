from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .obstacles3d import BoxObstacle, SphericalObstacle

ObstacleType = Tuple[str, object]

DEFAULT_LIMITS: Tuple[float, float] = (-15.0, 15.0)


@dataclass
class NeedleRobot3D:
    """Spherical needle tip used only for collision checking."""

    radius: float = 0.2


@dataclass
class World3D:
    """
    3D world for a steerable-needle robot: box bounds, static obstacles
    (walls + balls), and a voxel-based tumor field that injections heal.

    This mirrors `sim2d.world.World2D` but in three dimensions. Tumors live on a
    discrete voxel grid (so "heal within a radius" is a cheap masking op) and do
    NOT affect collision passability.
    """

    x_limits: Tuple[float, float] = DEFAULT_LIMITS
    y_limits: Tuple[float, float] = DEFAULT_LIMITS
    z_limits: Tuple[float, float] = DEFAULT_LIMITS
    robot: NeedleRobot3D = field(default_factory=NeedleRobot3D)
    state: np.ndarray = field(init=False)  # [x, y, z]
    orientation: int = 0
    obstacles: List[ObstacleType] = field(default_factory=list)
    goal: Optional[Tuple[float, float, float]] = None

    trail: List[Tuple[float, float, float]] = field(default_factory=list, init=False)
    retracted_edges: List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = field(default_factory=list, init=False)

    # Tumor voxel grid -------------------------------------------------
    # Cell size of 1.0 keeps voxel indexing aligned with the integer motion
    # lattice (step_size == 1), so path coordinates stay integers.
    tumor_cell_size: float = 1.0
    tumor_injection_radius: float = 2.5
    tumor_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    healed_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    # Healthy tissue the DP planner's damage objective tracks: every voxel that
    # is neither tumor nor obstacle.
    healthy_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    # Visualization-only: healthy voxels a replayed dosage choice damaged.
    # Has no effect on planning.
    damaged_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _x_centers: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _y_centers: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _z_centers: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.state = np.zeros(3, dtype=float)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_robot_state(self, x: float, y: float, z: float, orientation: int = 0) -> None:
        self.state = np.array([x, y, z], dtype=float)
        self.orientation = int(orientation)

    def add_spherical_obstacle(
        self, center: Tuple[float, float, float], radius: float
    ) -> SphericalObstacle:
        obs = SphericalObstacle(center=center, radius=radius)
        self.obstacles.append(("sphere", obs))
        return obs

    def add_box_obstacle(
        self,
        x: float,
        y: float,
        z: float,
        width: float,
        depth: float,
        height: float,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> BoxObstacle:
        obs = BoxObstacle(
            x=x, y=y, z=z, width=width, depth=depth, height=height,
            yaw=yaw, pitch=pitch, roll=roll,
        )
        self.obstacles.append(("box", obs))
        return obs

    def set_goal(self, gx: float, gy: float, gz: float) -> None:
        self.goal = (gx, gy, gz)

    def reset_trail(self) -> None:
        self.trail = []
        self.retracted_edges = []

    def append_trail_point(self, x: float, y: float, z: float) -> None:
        self.trail.append((float(x), float(y), float(z)))

    def retract_trail_step(self) -> Optional[Tuple[float, float, float]]:
        if len(self.trail) < 2:
            return None
        b = self.trail.pop()
        a = self.trail[-1]
        self.retracted_edges.append((a, b))
        return a

    # ------------------------------------------------------------------
    # Tumor voxel grid
    # ------------------------------------------------------------------
    def ensure_tumor_grid(self) -> None:
        if self.tumor_mask is not None:
            return
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        z_min, z_max = self.z_limits

        nx = max(1, int(np.round((x_max - x_min) / self.tumor_cell_size)))
        ny = max(1, int(np.round((y_max - y_min) / self.tumor_cell_size)))
        nz = max(1, int(np.round((z_max - z_min) / self.tumor_cell_size)))

        csx = (x_max - x_min) / nx
        csy = (y_max - y_min) / ny
        csz = (z_max - z_min) / nz

        self._x_centers = x_min + (np.arange(nx) + 0.5) * csx
        self._y_centers = y_min + (np.arange(ny) + 0.5) * csy
        self._z_centers = z_min + (np.arange(nz) + 0.5) * csz

        self.tumor_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.healed_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.healthy_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.damaged_mask = np.zeros((nx, ny, nz), dtype=bool)

    def reset_tumor_grid(self) -> None:
        self.tumor_mask = None
        self.healed_mask = None
        self.healthy_mask = None
        self.damaged_mask = None
        self._x_centers = None
        self._y_centers = None
        self._z_centers = None

    def compute_healthy_mask(self) -> None:
        """
        Mark healthy tissue = every voxel that is neither tumor nor obstacle. Call
        after all tumors/obstacles are added (it rasterizes obstacles onto the grid).
        """
        self.ensure_tumor_grid()
        assert self._x_centers is not None and self._y_centers is not None
        assert self._z_centers is not None and self.tumor_mask is not None

        xs = self._x_centers[:, None, None]
        ys = self._y_centers[None, :, None]
        zs = self._z_centers[None, None, :]
        obstacle_mask = np.zeros_like(self.tumor_mask)
        for kind, obs in self.obstacles:
            if kind == "sphere":
                cx, cy, cz = obs.center
                dx = xs - cx
                dy = ys - cy
                dz = zs - cz
                obstacle_mask |= (dx * dx + dy * dy + dz * dz) <= (obs.radius * obs.radius)
            elif kind == "box":
                cx, cy, cz = obs.centroid()
                dx = xs - cx
                dy = ys - cy
                dz = zs - cz
                # Rotate voxel coords into the box's local (unrotated) frame
                # via R.T (inverse of an orthonormal rotation), then apply
                # the axis-aligned half-extent test in that frame.
                rot = obs.rotation_matrix()
                lx = rot[0, 0] * dx + rot[1, 0] * dy + rot[2, 0] * dz
                ly = rot[0, 1] * dx + rot[1, 1] * dy + rot[2, 1] * dz
                lz = rot[0, 2] * dx + rot[1, 2] * dy + rot[2, 2] * dz
                hw, hd, hh = obs.width / 2.0, obs.depth / 2.0, obs.height / 2.0
                obstacle_mask |= (
                    (-hw <= lx) & (lx <= hw)
                    & (-hd <= ly) & (ly <= hd)
                    & (-hh <= lz) & (lz <= hh)
                )

        self.healthy_mask = (~self.tumor_mask) & (~obstacle_mask)

    def add_ball_tumor(
        self, center: Tuple[float, float, float], radius: float
    ) -> None:
        """Add a spherical (ball) tumor region (unhealed until injected)."""
        self.ensure_tumor_grid()
        assert self._x_centers is not None and self._y_centers is not None
        assert self._z_centers is not None and self.tumor_mask is not None
        cx, cy, cz = center
        dx = self._x_centers[:, None, None] - cx
        dy = self._y_centers[None, :, None] - cy
        dz = self._z_centers[None, None, :] - cz
        mask = (dx * dx + dy * dy + dz * dz) <= radius * radius
        self.tumor_mask |= mask

    def window_bounds(
        self, x: float, y: float, z: float, radius: float
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Per-axis (lo, hi) grid-index bounds of a bounding box containing every
        cell within `radius` of (x, y, z), clamped to the grid extent. This is
        a box, not the exact sphere -- callers still apply the spherical
        distance test within it -- but slicing to it before that test keeps
        per-node coverage/damage scans from touching the whole grid (tens of
        thousands of cells) when only a small region around the point matters.
        """
        self.ensure_tumor_grid()
        assert self._x_centers is not None and self._y_centers is not None
        assert self._z_centers is not None
        nx, ny, nz = len(self._x_centers), len(self._y_centers), len(self._z_centers)
        cs = float(self.tumor_cell_size)
        half_cells = int(np.ceil(float(radius) / cs)) + 1

        def _bounds(center: float, n: int, axis_min: float) -> Tuple[int, int]:
            idx0 = int(np.floor((center - axis_min) / cs))
            return max(0, idx0 - half_cells), min(n, idx0 + half_cells + 1)

        ix_lo, ix_hi = _bounds(x, nx, self.x_limits[0])
        iy_lo, iy_hi = _bounds(y, ny, self.y_limits[0])
        iz_lo, iz_hi = _bounds(z, nz, self.z_limits[0])
        return ix_lo, ix_hi, iy_lo, iy_hi, iz_lo, iz_hi

    def covered_pois_at(
        self, x: float, y: float, z: float, radius: Optional[float] = None
    ) -> "frozenset[Tuple[int, int, int]]":
        """
        The set of tumor-voxel POIs covered by an injection at (x, y, z).

        A POI is the integer voxel index (ix, iy, iz). This is the SINGLE source
        of truth for coverage: graph generation hands these sets to the planner
        (which treats them as opaque sets), and `inject_at` heals exactly this
        set. So "covered by IRIS" and "healed in the world" can never disagree.
        """
        self.ensure_tumor_grid()
        assert self._x_centers is not None and self._y_centers is not None
        assert self._z_centers is not None and self.tumor_mask is not None

        radius = self.tumor_injection_radius if radius is None else float(radius)
        ix_lo, ix_hi, iy_lo, iy_hi, iz_lo, iz_hi = self.window_bounds(x, y, z, radius)
        dx = self._x_centers[ix_lo:ix_hi, None, None] - x
        dy = self._y_centers[None, iy_lo:iy_hi, None] - y
        dz = self._z_centers[None, None, iz_lo:iz_hi] - z
        radius_mask = (dx * dx + dy * dy + dz * dz) <= radius * radius
        covered = radius_mask & self.tumor_mask[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        ix, iy, iz = np.nonzero(covered)
        return frozenset(
            zip((ix + ix_lo).tolist(), (iy + iy_lo).tolist(), (iz + iz_lo).tolist())
        )

    def heal_pois(self, pois) -> int:
        """
        Mark a set of voxel-index POIs as healed. Returns the count of POIs that
        were tumor voxels and not already healed.
        """
        self.ensure_tumor_grid()
        assert self.tumor_mask is not None and self.healed_mask is not None
        n = 0
        for ix, iy, iz in pois:
            if self.tumor_mask[ix, iy, iz] and not self.healed_mask[ix, iy, iz]:
                self.healed_mask[ix, iy, iz] = True
                n += 1
        return n

    def inject_at(
        self, x: float, y: float, z: float, radius: Optional[float] = None
    ) -> int:
        """Inject at (x, y, z): heal exactly the POIs `covered_pois_at` reports."""
        return self.heal_pois(self.covered_pois_at(x, y, z, radius))

    def mark_damaged_pois(self, pois) -> None:
        """
        Paint an explicit set of healthy-voxel POIs as damaged, for
        visualization only (mirrors `World2D.paint_injection_cells`'s damaged
        side) -- does not affect collision, tumor state, or planning.
        """
        self.ensure_tumor_grid()
        assert self.damaged_mask is not None
        for ix, iy, iz in pois:
            self.damaged_mask[ix, iy, iz] = True

    def damaged_voxel_centers(self) -> np.ndarray:
        """Return an (N, 3) array of damaged-voxel center coordinates."""
        self.ensure_tumor_grid()
        assert self.damaged_mask is not None
        assert self._x_centers is not None and self._y_centers is not None
        assert self._z_centers is not None
        ix, iy, iz = np.nonzero(self.damaged_mask)
        return np.stack(
            [self._x_centers[ix], self._y_centers[iy], self._z_centers[iz]], axis=1
        )

    def tumor_voxel_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (centers, healed_flags) for every tumor voxel.

        `centers` is an (N, 3) array of voxel-center coordinates; `healed_flags`
        is a length-N boolean array (True = already healed/green).
        """
        self.ensure_tumor_grid()
        assert self.tumor_mask is not None and self.healed_mask is not None
        assert self._x_centers is not None and self._y_centers is not None
        assert self._z_centers is not None
        ix, iy, iz = np.nonzero(self.tumor_mask)
        centers = np.stack(
            [self._x_centers[ix], self._y_centers[iy], self._z_centers[iz]], axis=1
        )
        healed = self.healed_mask[ix, iy, iz]
        return centers, healed

    # ------------------------------------------------------------------
    # Collision / bounds
    # ------------------------------------------------------------------
    def in_collision(self, point: Optional[np.ndarray] = None) -> bool:
        if point is None:
            point = self.state
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        margin = self.robot.radius

        if (
            x < self.x_limits[0] + margin
            or x > self.x_limits[1] - margin
            or y < self.y_limits[0] + margin
            or y > self.y_limits[1] - margin
            or z < self.z_limits[0] + margin
            or z > self.z_limits[1] - margin
        ):
            return True

        for _kind, obs in self.obstacles:
            if obs.collides((x, y, z), margin=margin):  # type: ignore[attr-defined]
                return True
        return False
