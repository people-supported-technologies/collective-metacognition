"""Build NetworkX graph from atomic edges keyed on canonical node IDs."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .schemas import CanonicalNode, EdgeRecord


def build_graph(
    edges: list[EdgeRecord],
    registry: dict[str, CanonicalNode],
) -> nx.DiGraph:
    """Build a directed graph with all registry nodes (fixed universe) and extracted edges."""
    G = nx.DiGraph()

    for node_id, node in registry.items():
        G.add_node(
            node_id,
            label=node.label,
            type=node.type.value,
            aliases=node.aliases,
        )

    for edge in edges:
        src = edge.source_node_id
        tgt = edge.target_node_id
        if src is None or tgt is None:
            continue

        edge_key = (src, tgt, edge.relation.value)
        if G.has_edge(src, tgt) and G[src][tgt].get("relation") == edge.relation.value:
            G[src][tgt]["edges"].append(edge)
        else:
            G.add_edge(
                src,
                tgt,
                relation=edge.relation.value,
                edges=[edge],
            )

    return G
