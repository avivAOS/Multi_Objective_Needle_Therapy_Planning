"""3D steerable-needle simulation environment.

Mirrors the 2D `sim2d` package in three dimensions and feeds the same
graph-search algorithms (the IRIS planner in `iris_on_graph.py`), which are
already generic over node id and coverage POI types.
"""

from __future__ import annotations

from .obstacles3d import BoxObstacle, SphericalObstacle, rotation_matrix
from .primitives3d import (
    DEFAULT_MAX_TURN_DEG,
    DIRECTIONS_3D,
    NUM_ORIENTATIONS,
    MotionPrimitive3D,
    allowed_primitives_3d,
    apply_primitive_3d,
    direction_from_vector,
    direction_vector,
    orientation_to_heading,
)
from .world3d import DEFAULT_LIMITS, NeedleRobot3D, World3D
from .demo_map3d import configure_demo_map_3d
from .paths_graph3d import (
    DOSAGE_LEVELS,
    dosage_to_radius_3d,
    compute_dosage_coverage_and_damage_for_xyz,
    dosage_coverage_3d,
    dosage_damage_3d,
    make_dosage_fn_3d,
    generate_paths_graph_3d,
    inject_coverage,
    make_inject_coverage_fn,
)
from .path3d import (
    Path3D,
    PathStep3D,
    apply_step,
    init_world_for_path,
    path3d_from_iris_graph_path,
)
from .render3d import highlight_obstacle, render_world_3d, set_axes_equal

__all__ = [
    "BoxObstacle",
    "SphericalObstacle",
    "rotation_matrix",
    "DEFAULT_MAX_TURN_DEG",
    "DIRECTIONS_3D",
    "NUM_ORIENTATIONS",
    "MotionPrimitive3D",
    "allowed_primitives_3d",
    "apply_primitive_3d",
    "direction_from_vector",
    "direction_vector",
    "orientation_to_heading",
    "DEFAULT_LIMITS",
    "NeedleRobot3D",
    "World3D",
    "configure_demo_map_3d",
    "DOSAGE_LEVELS",
    "dosage_to_radius_3d",
    "compute_dosage_coverage_and_damage_for_xyz",
    "dosage_coverage_3d",
    "dosage_damage_3d",
    "make_dosage_fn_3d",
    "generate_paths_graph_3d",
    "inject_coverage",
    "make_inject_coverage_fn",
    "Path3D",
    "PathStep3D",
    "apply_step",
    "init_world_for_path",
    "path3d_from_iris_graph_path",
    "render_world_3d",
    "set_axes_equal",
    "highlight_obstacle",
]