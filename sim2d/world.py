from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .obstacles import CircularObstacle, RectangularObstacle
from .robot import DifferentialDriveRobot
from .viewport import DEFAULT_VIEW_X_LIMITS, DEFAULT_VIEW_Y_LIMITS


ObstacleType = Tuple[str, object]


@dataclass
class World2D:
    """
    2D world with a simple circular robot and static obstacles.

    Coordinates are continuous, but robot motion is expected to happen
    in discrete steps defined by motion primitives (see `primitives.py`).
    """

    x_limits: Tuple[float, float] = DEFAULT_VIEW_X_LIMITS
    y_limits: Tuple[float, float] = DEFAULT_VIEW_Y_LIMITS
    robot: DifferentialDriveRobot = field(default_factory=DifferentialDriveRobot)
    state: np.ndarray = field(init=False)
    obstacles: List[ObstacleType] = field(default_factory=list)
    goal: Optional[Tuple[float, float]] = None
    # Active trail is the "current path" after any retractions.
    trail: List[Tuple[float, float]] = field(default_factory=list, init=False)
    # Edges that were traversed but later retracted (rendered grey).
    retracted_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list, init=False
    )

    # ------------------------------------------------------------------
    # Tumor grid (yellow/green/purple)
    # ------------------------------------------------------------------
    # Tumors are rendered/updated on a discrete grid for easy "paint within
    # radius" operations. Grid does NOT affect collision passability.
    tumor_cell_size: float = 0.05
    tumor_injection_radius: float = 0.75
    tumor_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    healed_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    damaged_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _x_centers: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _y_centers: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _tumor_grid_x_limits: Optional[Tuple[float, float]] = field(
        init=False, default=None, repr=False
    )
    _tumor_grid_y_limits: Optional[Tuple[float, float]] = field(
        init=False, default=None, repr=False
    )

    # Healthy/sensitive tissue the planner's damage objective tracks. Shares the
    # tumor grid (1:1 cells, so bitset indices line up). Distinct from damaged_mask,
    # which is only a visualization of what an injection happened to touch.
    healthy_mask: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.state = np.zeros(3, dtype=float)

    # ------------------------------------------------------------------
    # World configuration
    # ------------------------------------------------------------------
    def set_robot_state(self, x: float, y: float, theta: float) -> None:
        self.state = np.array([x, y, theta], dtype=float)

    def add_circular_obstacle(
        self, center: Tuple[float, float], radius: float
    ) -> CircularObstacle:
        obs = CircularObstacle(center=center, radius=radius)
        self.obstacles.append(("circle", obs))
        return obs

    def add_rectangular_obstacle(
        self, x: float, y: float, width: float, height: float
    ) -> RectangularObstacle:
        obs = RectangularObstacle(x=x, y=y, width=width, height=height)
        self.obstacles.append(("rect", obs))
        return obs

    def set_goal(self, gx: float, gy: float) -> None:
        self.goal = (gx, gy)

    def reset_trail(self) -> None:
        """Clear the recorded trail points."""
        self.trail = []
        self.retracted_edges = []

    def append_trail_point(self, x: float, y: float) -> None:
        """Record a point as having been visited by the robot."""
        self.trail.append((float(x), float(y)))

    def retract_trail_step(self) -> Optional[Tuple[float, float]]:
        """
        Step back to the previous trail pose (the edge moves to `retracted_edges`
        for grey rendering). Returns the new (x, y), or None if the trail is too short.
        """
        if len(self.trail) < 2:
            return None
        b = self.trail.pop()
        a = self.trail[-1]
        self.retracted_edges.append((a, b))
        return a

    # ------------------------------------------------------------------
    # Tumors (yellow/green/purple) on a discrete grid
    # ------------------------------------------------------------------
    def ensure_tumor_grid(self) -> None:
        """
        Create tumor grid masks based on current `x_limits/y_limits`.

        Grid cell values:
        - `tumor_mask`: True => yellow (unless healed_mask is also True)
        - `healed_mask`: True => green (overrides yellow)
        - `damaged_mask`: True => purple (only for non-tumor cells)
        """
        if self.tumor_mask is not None:
            return

        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        self._tumor_grid_x_limits = (float(x_min), float(x_max))
        self._tumor_grid_y_limits = (float(y_min), float(y_max))

        nx = max(1, int(np.round((x_max - x_min) / self.tumor_cell_size)))
        ny = max(1, int(np.round((y_max - y_min) / self.tumor_cell_size)))

        # Use x_limits/y_limits as the render extents; compute actual spacing
        # from nx/ny so that centers sit evenly within those extents.
        cell_size_x = (x_max - x_min) / nx
        cell_size_y = (y_max - y_min) / ny

        self._x_centers = x_min + (np.arange(nx) + 0.5) * cell_size_x
        self._y_centers = y_min + (np.arange(ny) + 0.5) * cell_size_y

        self.tumor_mask = np.zeros((ny, nx), dtype=bool)
        self.healed_mask = np.zeros((ny, nx), dtype=bool)
        self.damaged_mask = np.zeros((ny, nx), dtype=bool)
        self.healthy_mask = np.zeros((ny, nx), dtype=bool)

    def reset_injection_paint(self) -> None:
        """Clear healed/purple paint but keep the tumor/healthy masks and grid
        (so a path can be replayed without rebuilding the map)."""
        if self.healed_mask is not None:
            self.healed_mask[:] = False
        if self.damaged_mask is not None:
            self.damaged_mask[:] = False

    def reset_tumor_grid(self) -> None:
        """Drop all masks and cached grid centers so the grid rebuilds (e.g. after
        the viewport limits change)."""
        self.tumor_mask = None
        self.healed_mask = None
        self.damaged_mask = None
        self.healthy_mask = None
        self._x_centers = None
        self._y_centers = None
        self._tumor_grid_x_limits = None
        self._tumor_grid_y_limits = None

    def add_circular_tumor(
        self, center: Tuple[float, float], radius: float
    ) -> None:
        """Add a circular tumor region (painted yellow until healed)."""
        self.ensure_tumor_grid()
        assert self._x_centers is not None
        assert self._y_centers is not None
        cx, cy = center

        dx = self._x_centers[None, :] - cx
        dy = self._y_centers[:, None] - cy
        mask = (dx * dx + dy * dy) <= radius * radius
        self.tumor_mask |= mask

    def add_rectangular_tumor(
        self, x: float, y: float, width: float, height: float
    ) -> None:
        """Add an axis-aligned rectangular tumor region (yellow until healed)."""
        self.ensure_tumor_grid()
        assert self._x_centers is not None
        assert self._y_centers is not None

        mask = (
            (x <= self._x_centers[None, :])
            & (self._x_centers[None, :] <= x + width)
            & (y <= self._y_centers[:, None])
            & (self._y_centers[:, None] <= y + height)
        )
        self.tumor_mask |= mask

    def compute_healthy_mask(self) -> None:
        """
        Mark healthy tissue = every cell that is neither tumor nor obstacle. Call
        after all tumors/obstacles are added (it rasterizes obstacles onto the grid).
        """
        self.ensure_tumor_grid()
        assert self._x_centers is not None
        assert self._y_centers is not None
        assert self.tumor_mask is not None

        xs = self._x_centers[None, :]
        ys = self._y_centers[:, None]
        obstacle_mask = np.zeros_like(self.tumor_mask)
        for kind, obs in self.obstacles:
            if kind == "circle":
                cx, cy = obs.center
                dx = xs - cx
                dy = ys - cy
                obstacle_mask |= (dx * dx + dy * dy) <= (obs.radius * obs.radius)
            elif kind == "rect":
                obstacle_mask |= (
                    (obs.x <= xs)
                    & (xs <= obs.x + obs.width)
                    & (obs.y <= ys)
                    & (ys <= obs.y + obs.height)
                )

        self.healthy_mask = (~self.tumor_mask) & (~obstacle_mask)

    def inject_at(self, x: float, y: float, radius: Optional[float] = None) -> None:
        """
        Inject at robot position (x, y).

        Rules:
        - Heal tumor points within `radius` (yellow -> green).
        - Paint damaged all non-tumor points within `radius`.
        - Points occluded by an obstacle (a wall between the injection site and
          the point) are excluded from both effects.
        - Tumor effects do NOT affect collision passability.
        - Damaged/healed paint persists for the rest of the simulation.
        """
        self.ensure_tumor_grid()
        assert self._x_centers is not None
        assert self._y_centers is not None
        assert self.tumor_mask is not None
        assert self.healed_mask is not None
        assert self.damaged_mask is not None

        radius = self.tumor_injection_radius if radius is None else float(radius)
        dx = self._x_centers[None, :] - x
        dy = self._y_centers[:, None] - y
        radius_mask = (dx * dx + dy * dy) <= radius * radius

        if self.obstacles:
            iy_idx, ix_idx = np.nonzero(radius_mask)
            for iy, ix in zip(iy_idx, ix_idx):
                cx = float(self._x_centers[ix])
                cy = float(self._y_centers[iy])
                if self.line_of_sight_blocked((x, y), (cx, cy)):
                    radius_mask[iy, ix] = False

        heal_targets = radius_mask & self.tumor_mask & ~self.healed_mask
        if np.any(heal_targets):
            self.healed_mask[heal_targets] = True
            # Safety: healed tumor cells must never be purple.
            self.damaged_mask[heal_targets] = False

        purple_targets = radius_mask & ~self.tumor_mask
        if np.any(purple_targets):
            self.damaged_mask[purple_targets] = True

    def _center_to_index(self, cx: float, cy: float) -> Optional[Tuple[int, int]]:
        """Map a grid-cell *center* (cx, cy) back to its (iy, ix) indices."""
        if (
            self._x_centers is None
            or self._y_centers is None
            or self._tumor_grid_x_limits is None
            or self._tumor_grid_y_limits is None
        ):
            return None
        x_min, x_max = self._tumor_grid_x_limits
        y_min, y_max = self._tumor_grid_y_limits
        nx = int(self._x_centers.shape[0])
        ny = int(self._y_centers.shape[0])
        csx = (float(x_max) - float(x_min)) / nx
        csy = (float(y_max) - float(y_min)) / ny
        if csx <= 0.0 or csy <= 0.0:
            return None
        ix = int(round((float(cx) - float(x_min)) / csx - 0.5))
        iy = int(round((float(cy) - float(y_min)) / csy - 0.5))
        if 0 <= ix < nx and 0 <= iy < ny:
            return iy, ix
        return None

    def paint_injection_cells(
        self,
        covered_centers: Iterable[Tuple[float, float]] = (),
        damaged_centers: Iterable[Tuple[float, float]] = (),
    ) -> None:
        """
        Paint an explicit set of cells (green=covered, purple=damaged) instead of a
        fixed radius, so the visualization matches exactly what a planner counted
        (the DP's per-dose coverage/damage, which `inject_at`'s single radius can't
        reproduce).
        """
        self.ensure_tumor_grid()
        assert self.healed_mask is not None
        assert self.damaged_mask is not None
        assert self.tumor_mask is not None
        for cx, cy in covered_centers:
            idx = self._center_to_index(cx, cy)
            if idx is not None:
                iy, ix = idx
                self.healed_mask[iy, ix] = True
                self.damaged_mask[iy, ix] = False
        for cx, cy in damaged_centers:
            idx = self._center_to_index(cx, cy)
            if idx is not None:
                iy, ix = idx
                if not self.tumor_mask[iy, ix]:
                    self.damaged_mask[iy, ix] = True

    # ------------------------------------------------------------------
    # Collision and bounds checking
    # ------------------------------------------------------------------
    def in_collision(self, state: Optional[np.ndarray] = None) -> bool:
        if state is None:
            state = self.state
        x, y, _ = state

        # World bounds
        if (
            x < self.x_limits[0] + self.robot.radius
            or x > self.x_limits[1] - self.robot.radius
            or y < self.y_limits[0] + self.robot.radius
            or y > self.y_limits[1] - self.robot.radius
        ):
            return True

        # Obstacles
        for kind, obs in self.obstacles:
            if kind == "circle" and obs.collides((x, y), margin=self.robot.radius):
                return True
            if kind == "rect" and obs.collides((x, y), margin=self.robot.radius):
                return True

        return False

    def line_of_sight_blocked(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        """True if any obstacle sits between p0 and p1 (injections don't reach through walls)."""
        return any(obs.blocks_segment(p0, p1) for _kind, obs in self.obstacles)

    def distance_to_goal(self) -> Optional[float]:
        if self.goal is None:
            return None
        gx, gy = self.goal
        x, y, _ = self.state
        return float(np.hypot(gx - x, gy - y))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(
        self,
        ax: Optional[plt.Axes] = None,
        clear: bool = True,
        *,
        show_robot: bool = True,
        show_trail: bool = True,
        show_goal: bool = True,
        show_healthy_overlay: bool = False,
    ) -> plt.Axes:
        """Draw the world (obstacles, tumor grid, trail, robot, goal) onto `ax`."""
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        if clear:
            ax.cla()

        ax.set_xlim(*self.x_limits)
        ax.set_ylim(*self.y_limits)
        ax.set_aspect("equal", "box")

        # Faint tint over at-risk healthy tissue (static map renders only; during
        # animation the purple paint already shows what was damaged).
        if (
            show_healthy_overlay
            and self.healthy_mask is not None
            and self._x_centers is not None
            and self._y_centers is not None
            and self._tumor_grid_x_limits is not None
            and self._tumor_grid_y_limits is not None
        ):
            ny, nx = self.healthy_mask.shape
            healthy_overlay_img = np.zeros((ny, nx, 4), dtype=float)
            healthy_overlay_img[self.healthy_mask, 0] = 0.53
            healthy_overlay_img[self.healthy_mask, 1] = 0.81
            healthy_overlay_img[self.healthy_mask, 2] = 0.98
            healthy_overlay_img[self.healthy_mask, 3] = 0.22
            x_min, x_max = self._tumor_grid_x_limits
            y_min, y_max = self._tumor_grid_y_limits
            ax.imshow(
                healthy_overlay_img,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                interpolation="nearest",
                zorder=0,
            )

        # Draw tumor grid (yellow/green/purple) behind obstacles.
        if (
            self.tumor_mask is not None
            and self.healed_mask is not None
            and self.damaged_mask is not None
            and self._x_centers is not None
            and self._y_centers is not None
            and self._tumor_grid_x_limits is not None
            and self._tumor_grid_y_limits is not None
        ):
            ny, nx = self.tumor_mask.shape
            img = np.zeros((ny, nx, 4), dtype=float)
            alpha = 0.35

            yellow = self.tumor_mask & ~self.healed_mask
            green = self.healed_mask
            purple = self.damaged_mask & ~self.tumor_mask
            # healthy_mask is not painted (it's almost the whole workspace); purple
            # already shows actual off-target damage.

            if np.any(yellow):
                img[yellow, 0] = 1.0
                img[yellow, 1] = 1.0
                img[yellow, 2] = 0.0
                img[yellow, 3] = alpha

            if np.any(green):
                img[green, 0] = 0.0
                img[green, 1] = 1.0
                img[green, 2] = 0.0
                img[green, 3] = alpha

            if np.any(purple):
                img[purple, 0] = 0.5
                img[purple, 1] = 0.0
                img[purple, 2] = 0.5
                img[purple, 3] = alpha

            x_min, x_max = self._tumor_grid_x_limits
            y_min, y_max = self._tumor_grid_y_limits
            ax.imshow(
                img,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                interpolation="nearest",
                zorder=0,
            )

        for kind, obs in self.obstacles:
            if kind == "circle":
                cx, cy = obs.center
                circle = plt.Circle(
                    (cx, cy),
                    obs.radius,
                    color="tab:red",
                    alpha=0.5,
                )
                ax.add_patch(circle)
            elif kind == "rect":
                rect = plt.Rectangle(
                    (obs.x, obs.y),
                    obs.width,
                    obs.height,
                    color="tab:red",
                    alpha=0.5,
                )
                ax.add_patch(rect)

        if show_trail:
            # Draw traveled path (behind robot).
            if self.retracted_edges:
                for (a, b) in self.retracted_edges:
                    ax.plot(
                        [a[0], b[0]],
                        [a[1], b[1]],
                        color="0.6",
                        linewidth=2,
                        alpha=0.6,
                        zorder=1,
                    )

            if len(self.trail) >= 2:
                xs, ys = zip(*self.trail)
                ax.plot(xs, ys, color="tab:blue", linewidth=2, alpha=0.35, zorder=2)

        if show_robot:
            x, y, theta = self.state
            body = plt.Circle(
                (x, y),
                self.robot.radius,
                color="tab:blue",
                alpha=0.8,
                zorder=3,
            )
            ax.add_patch(body)
            hx = x + self.robot.radius * np.cos(theta)
            hy = y + self.robot.radius * np.sin(theta)
            ax.plot([x, hx], [y, hy], color="k", zorder=4)

        if show_goal and self.goal is not None:
            gx, gy = self.goal
            goal = plt.Circle(
                (gx, gy),
                0.1,
                color="tab:green",
            )
            ax.add_patch(goal)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if created_fig:
            plt.tight_layout()

        return ax

