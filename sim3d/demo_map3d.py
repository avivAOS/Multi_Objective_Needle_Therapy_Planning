from __future__ import annotations

from .world3d import World3D


def configure_demo_map_3d(
    world: World3D,
    *,
    start_x: float,
    start_y: float,
    start_z: float,
    add_obstacles: bool = True,
    add_tumors: bool = True,
    add_healthy: bool = True,
) -> None:
    if add_obstacles:
        world.add_box_obstacle(x=0.5035, y=0.8345, z=0.669, width=3.993, depth=1.331, height=2.662, yaw=0.0, pitch=0.0, roll=0.0)
        world.add_box_obstacle(x=9.5, y=-1.5, z=0.0, width=6.0, depth=2.0, height=2.0, yaw=0.0, pitch=0.0, roll=0.0)

    if not add_tumors:
        return

    world.reset_tumor_grid()

    world.add_ball_tumor(center=(6.0, 6.0, 6.0), radius=1.8182)
    world.add_ball_tumor(center=(10.0, 3.0, 5.0), radius=2.0)

    # Healthy tissue (DP damage objective) = every non-tumor,
    # non-obstacle voxel -- must be computed after tumors are placed.
    if add_healthy:
        world.compute_healthy_mask()
