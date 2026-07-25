from __future__ import annotations

from .world import World2D


def configure_demo_map(
    world: World2D,
    *,
    start_x: float,
    start_y: float,
    add_obstacles: bool = True,
    add_tumors: bool = True,
    add_healthy: bool = True,
) -> None:
    world.x_limits = (-5.0, 5.0)
    world.y_limits = (0.0, 10.0)

    if add_obstacles:
        world.add_rectangular_obstacle(x=-5.0, y=4.7451, width=0.1569, height=2.8366)
        world.add_rectangular_obstacle(x=-5.0, y=4.7451, width=2.2745, height=0.3268)
        world.add_rectangular_obstacle(x=-2.6863, y=4.7451, width=0.183, height=1.9216)
        world.add_rectangular_obstacle(x=2.3072, y=5.0588, width=0.1569, height=2.4314)
        world.add_rectangular_obstacle(x=2.4118, y=5.098, width=2.5621, height=0.2222)
        world.add_rectangular_obstacle(x=4.8431, y=4.9804, width=0.1569, height=3.3987)

    if not add_tumors:
        return

    world.reset_tumor_grid()

    world.add_circular_tumor(center=(-3.7712, 5.9608), radius=0.7712)
    world.add_circular_tumor(center=(3.7708, 6.3399), radius=0.8109)
    world.add_circular_tumor(center=(-3.7577, 7.0322), radius=0.7586)
    world.add_circular_tumor(center=(3.5854, 7.5163), radius=0.761)
    world.add_circular_tumor(center=(-0.4052, 8.5137), radius=0.8327)

    # Healthy tissue (DP damage objective) = every non-tumor,
    # non-obstacle cell -- must be computed after tumors are placed.
    if add_healthy:
        world.compute_healthy_mask()
