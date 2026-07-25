from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .obstacles3d import BoxObstacle, SphericalObstacle
from .primitives3d import orientation_to_heading
from .world3d import World3D

# Cube face vertex indices into BoxObstacle.corners() (see obstacles3d.py).
_BOX_FACES = (
    (0, 1, 2, 3),  # bottom (z = z0)
    (4, 5, 6, 7),  # top    (z = z1)
    (0, 1, 5, 4),  # y = y0
    (2, 3, 7, 6),  # y = y1
    (1, 2, 6, 5),  # x = x1
    (0, 3, 7, 4),  # x = x0
)

# The 12 edges of a box, as index pairs into BoxObstacle.corners().
_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def _draw_box(ax, box: BoxObstacle, *, color: str = "tab:red", alpha: float = 0.25) -> None:
    corners = box.corners()
    faces = [[corners[i] for i in face] for face in _BOX_FACES]
    coll = Poly3DCollection(faces, facecolor=color, edgecolor="0.4", alpha=alpha)
    ax.add_collection3d(coll)


def _draw_sphere(
    ax, sphere: SphericalObstacle, *, color: str = "tab:red", alpha: float = 0.2, res: int = 14
) -> None:
    cx, cy, cz = sphere.center
    r = sphere.radius
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def set_axes_equal(ax, world: World3D) -> None:
    ax.set_xlim(*world.x_limits)
    ax.set_ylim(*world.y_limits)
    ax.set_zlim(*world.z_limits)
    try:
        ax.set_box_aspect(
            (
                world.x_limits[1] - world.x_limits[0],
                world.y_limits[1] - world.y_limits[0],
                world.z_limits[1] - world.z_limits[0],
            )
        )
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def render_world_3d(
    ax,
    world: World3D,
    *,
    clear: bool = True,
    show_obstacles: bool = True,
    show_tumors: bool = True,
    show_damage: bool = True,
    show_robot: bool = True,
    show_trail: bool = True,
    robot_pos: Optional[Sequence[float]] = None,
    tumor_marker_size: float = 8.0,
) -> None:
    """Render obstacles, tumor voxels, damaged (healthy) voxels, the traveled
    trail and the needle tip."""
    if clear:
        ax.cla()
    set_axes_equal(ax, world)

    if show_obstacles:
        for kind, obs in world.obstacles:
            if kind == "box":
                _draw_box(ax, obs)  # type: ignore[arg-type]
            elif kind == "sphere":
                _draw_sphere(ax, obs)  # type: ignore[arg-type]

    if show_tumors:
        centers, healed = world.tumor_voxel_centers()
        if centers.size:
            unhealed = ~healed
            if np.any(unhealed):
                pts = centers[unhealed]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c="gold", s=tumor_marker_size, alpha=0.5, depthshade=True,
                )
            if np.any(healed):
                pts = centers[healed]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c="limegreen", s=tumor_marker_size, alpha=0.7, depthshade=True,
                )

    if show_damage and world.damaged_mask is not None:
        damaged = world.damaged_voxel_centers()
        if damaged.size:
            ax.scatter(
                damaged[:, 0], damaged[:, 1], damaged[:, 2],
                c="purple", s=tumor_marker_size, alpha=0.6, depthshade=True,
            )

    if show_trail:
        for a, b in world.retracted_edges:
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="0.6", linewidth=2, alpha=0.6,
            )
        if len(world.trail) >= 2:
            tr = np.asarray(world.trail, dtype=float)
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="tab:blue", linewidth=2.5, alpha=0.9)

    if show_robot:
        pos = world.state if robot_pos is None else np.asarray(robot_pos, dtype=float)
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            c="navy", s=60, marker="o", depthshade=False,
        )
        if robot_pos is None:
            # world.orientation is only meaningful for the real, tracked robot
            # state (set via world.set_robot_state each step) -- an explicit
            # robot_pos override (e.g. the map editor's start-position preview)
            # has no matching orientation here, so skip the heading arrow.
            hx, hy, hz = orientation_to_heading(world.orientation)
            heading_len = 0.05 * (world.x_limits[1] - world.x_limits[0])
            ax.quiver(
                pos[0], pos[1], pos[2], hx, hy, hz,
                length=heading_len, color="black", linewidth=1.5,
                arrow_length_ratio=0.4,
            )

    if world.goal is not None:
        gx, gy, gz = world.goal
        ax.scatter([gx], [gy], [gz], c="black", s=80, marker="*")


def highlight_obstacle(
    ax,
    kind: str,
    obs: object,
    *,
    color: str = "#00e5ff",
    center: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
) -> None:
    """
    Draw a bright outline around the given obstacle to mark it as the
    map editor's currently-selected shape for interactive manipulation.

    For `kind == "box"`, `obs` must be a `BoxObstacle` (its rotated
    `corners()` are used, so the highlight follows the box's tilt). For
    `kind in ("sphere", "tumor")`, pass either a `SphericalObstacle` as
    `obs`, or `center`/`radius` directly (tumors aren't `SphericalObstacle`
    instances, so callers pass whatever's convenient).
    """
    if kind == "box":
        corners = obs.corners()  # type: ignore[attr-defined]
        segments = [[corners[a], corners[b]] for a, b in _BOX_EDGES]
        ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=2.5))
        return

    if center is None or radius is None:
        cx, cy, cz = obs.center  # type: ignore[attr-defined]
        r = obs.radius  # type: ignore[attr-defined]
    else:
        cx, cy, cz = center
        r = radius
    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 10)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, linewidth=1.0, alpha=0.9)
