from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Iterable, List, Mapping, Sequence, Tuple, TypeVar

NodeId = TypeVar("NodeId", bound=Hashable)


@dataclass(frozen=True)
class Graph(Generic[NodeId]):
    """Minimal directed graph (simulation-independent)."""

    nodes: Tuple[NodeId, ...]
    adjacency: Mapping[NodeId, Sequence[NodeId]]
    directed: bool = True

    def __contains__(self, node: NodeId) -> bool:
        return node in set(self.nodes)


@dataclass(frozen=True)
class GraphPathStep(Generic[NodeId]):
    step_index: int
    action: str  # "move" | "retract" | "inject" | "stay" | "restart"
    node: NodeId
    previous_node: NodeId | None
    # Post-move node; set for action=="move" so replay can resolve geometry without guessing.
    next_node: NodeId | None = None
    # Dosage level for action=="inject" steps produced by the DP planner.
    # None for legacy IRIS-produced inject steps (binary inject, no dosage concept).
    dosage: int | None = None


@dataclass(frozen=True)
class GraphPath(Generic[NodeId]):
    seed: int
    start_node: NodeId
    num_steps: int
    steps: Tuple[GraphPathStep[NodeId], ...]

    @property
    def nodes_visited(self) -> Tuple[NodeId, ...]:
        return tuple(s.node for s in self.steps)


def build_adjacency(
    edges: Iterable[Tuple[NodeId, NodeId]],
    *,
    directed: bool = True,
) -> Dict[NodeId, List[NodeId]]:
    adj: Dict[NodeId, List[NodeId]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        if not directed:
            adj.setdefault(v, []).append(u)
    return adj


def graph_from_edges(
    *,
    nodes: Iterable[NodeId],
    edges: Iterable[Tuple[NodeId, NodeId]],
    directed: bool = True,
) -> Graph[NodeId]:
    node_tuple = tuple(nodes)
    return Graph(
        nodes=node_tuple,
        adjacency=build_adjacency(edges, directed=directed),
        directed=bool(directed),
    )
