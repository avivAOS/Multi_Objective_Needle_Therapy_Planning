from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

from .world3d import World3D

Node3D = Tuple[float, float, float, int]


@dataclass
class PathStep3D:
    step_index: int
    action: str  # "move" | "inject" | "retract" | ...
    x: float
    y: float
    z: float
    ori: int
    is_inject: bool = False
    # Dosage level chosen at this inject step (DP planner only; None for
    # legacy binary IRIS/random injects).
    dosage: Optional[int] = None
    # POIs (voxel indices) healed at this inject step. This is the exact set the
    # planner credited for coverage, so replay heals precisely what the planner counted.
    covered_pois: Optional[List[Tuple[int, int, int]]] = None
    # POIs (healthy voxel indices) damaged at this inject step (DP planner
    # only; visualization only, mirrors `covered_pois`).
    damaged_pois: Optional[List[Tuple[int, int, int]]] = None


@dataclass
class Path3D:
    seed: int
    num_steps: int
    start_x: float
    start_y: float
    start_z: float
    start_orientation: int
    step_size: float
    # Size of the POI universe (coverable tumor voxels across all graph nodes),
    # i.e. the denominator IRIS reports coverage against.
    total_pois: int = 0
    steps: List[PathStep3D] = field(default_factory=list)


def path3d_from_iris_graph_path(
    *,
    graph_path,
    seed: int,
    step_size: float,
    nodes: Mapping[Node3D, dict],
) -> Path3D:
    """
    Convert an IRIS `GraphPath` (move/inject/retract steps over 3D node keys)
    into a `Path3D` for 3D replay. Mirrors the 2D converter in `main_demo.py`.
    """
    start_key = graph_path.start_node
    first = nodes[start_key]

    steps: List[PathStep3D] = []
    for i, st in enumerate(graph_path.steps):
        action = str(st.action)
        pre_key = st.node
        if action == "move":
            post_key = st.next_node
            if post_key is None:
                raise ValueError("IRIS move step missing next_node.")
        elif action == "retract":
            post_key = st.previous_node
            if post_key is None:
                raise ValueError("IRIS retract step missing previous_node.")
        else:
            post_key = pre_key

        post = nodes[post_key]
        covered_pois = None
        damaged_pois = None
        dosage = getattr(st, "dosage", None)
        if action == "inject":
            # The injection happens at the pre-move node; credit its POI set.
            pre_node = nodes[pre_key]
            if dosage is not None:
                # DP-produced step: dosage-indexed coverage/damage.
                from .paths_graph3d import dosage_coverage_3d, dosage_damage_3d

                covered_pois = [tuple(p) for p in dosage_coverage_3d(pre_node, dosage)]
                damaged_pois = [tuple(p) for p in dosage_damage_3d(pre_node, dosage)]
            else:
                # Legacy binary inject (IRIS/random): single-radius coverage only.
                covered_pois = [tuple(p) for p in pre_node.get("inject_coverage", ())]
        steps.append(
            PathStep3D(
                step_index=i,
                action=action,
                x=float(post["x"]),
                y=float(post["y"]),
                z=float(post["z"]),
                ori=int(post["ori"]),
                is_inject=(action == "inject"),
                dosage=dosage,
                covered_pois=covered_pois,
                damaged_pois=damaged_pois,
            )
        )

    universe: set = set()
    for node in nodes.values():
        universe |= set(node.get("inject_coverage", ()))

    return Path3D(
        seed=int(seed),
        num_steps=len(steps),
        start_x=float(first["x"]),
        start_y=float(first["y"]),
        start_z=float(first["z"]),
        start_orientation=int(first["ori"]),
        step_size=float(step_size),
        total_pois=len(universe),
        steps=steps,
    )


def init_world_for_path(world: World3D, path: Path3D) -> None:
    """Reset the world's trail/robot to the path's starting pose."""
    world.reset_trail()
    world.set_robot_state(path.start_x, path.start_y, path.start_z, path.start_orientation)
    world.append_trail_point(path.start_x, path.start_y, path.start_z)


def apply_step(world: World3D, step: PathStep3D) -> Optional[int]:
    """
    Advance the world by one path step (mutates trail/robot/tumor masks).

    Returns the number of voxels healed on an inject step, else None.
    """
    world.set_robot_state(step.x, step.y, step.z, step.ori)
    if step.is_inject:
        # Heal exactly the POIs the planner credited (not a re-derived geometry).
        if step.damaged_pois:
            world.mark_damaged_pois(step.damaged_pois)
        return world.heal_pois(step.covered_pois or ())
    if step.action == "retract":
        world.retract_trail_step()
        return None
    world.append_trail_point(step.x, step.y, step.z)
    return None
