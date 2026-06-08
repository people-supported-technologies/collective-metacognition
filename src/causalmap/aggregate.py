"""Aggregate-late module: compute per-filter statistics over the fixed node set.

All nodes from the registry are always present; a node with no edges under
the active filter has weight 0 (isolated, not removed).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schemas import CanonicalNode, EdgeRecord


def aggregate_edges(
    edges: list[EdgeRecord],
    registry: dict[str, CanonicalNode],
    filter_fn: callable | None = None,
) -> dict[str, Any]:
    """Aggregate edges applying an optional filter function.

    Returns a structure with:
      - nodes: all registry nodes with computed weights
      - edges: aggregated edge records with statistics
    """
    if filter_fn:
        filtered_edges = [e for e in edges if filter_fn(e)]
    else:
        filtered_edges = edges

    # Compute node weights (mention count under filter)
    node_weights: dict[str, int] = {nid: 0 for nid in registry}
    for edge in filtered_edges:
        if edge.source_node_id:
            node_weights[edge.source_node_id] = node_weights.get(edge.source_node_id, 0) + 1
        if edge.target_node_id:
            node_weights[edge.target_node_id] = node_weights.get(edge.target_node_id, 0) + 1

    # Aggregate edges by (source, target, relation)
    edge_groups: dict[tuple, list[EdgeRecord]] = defaultdict(list)
    for edge in filtered_edges:
        key = (edge.source_node_id, edge.target_node_id, edge.relation.value)
        edge_groups[key].append(edge)

    aggregated_edges: list[dict[str, Any]] = []
    for (src, tgt, rel), group in edge_groups.items():
        speakers = set(e.participant_id for e in group)
        tables = set(e.table_id for e in group)
        rounds = set(e.round_id for e in group)

        stance_counts: dict[str, int] = defaultdict(int)
        explicitness_counts: dict[str, int] = defaultdict(int)
        confidences: list[float] = []

        for e in group:
            stance_counts[e.stance.value] += 1
            explicitness_counts[e.explicitness.value] += 1
            confidences.append(e.confidence)

        total = len(group)
        stance_dist = {k: v / total for k, v in stance_counts.items()}
        expl_dist = {k: v / total for k, v in explicitness_counts.items()}

        aggregated_edges.append({
            "source_node_id": src,
            "target_node_id": tgt,
            "relation": rel,
            "mention_count": total,
            "unique_speakers": len(speakers),
            "unique_tables": len(tables),
            "unique_rounds": len(rounds),
            "stance_distribution": stance_dist,
            "explicitness_distribution": expl_dist,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "evidence_samples": [e.evidence_text for e in group[:5]],
            "speaker_ids": list(speakers),
        })

    nodes = []
    for nid, node in registry.items():
        nodes.append({
            "node_id": nid,
            "label": node.label,
            "type": node.type.value,
            "aliases": node.aliases,
            "weight": node_weights.get(nid, 0),
        })

    return {
        "nodes": nodes,
        "edges": aggregated_edges,
        "total_edges_raw": len(filtered_edges),
        "total_nodes": len(registry),
    }
