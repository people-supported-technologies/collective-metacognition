"""Aggregate-late module: compute per-filter statistics over the fixed node set.

All nodes from the registry are always present; a node with no edges under
the active filter has weight 0 (isolated, not removed). This module also computes
the signed polarity overlay (participant -> concept stances) and per-group
divergence, which is what makes demographic comparison visible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .ontology import polarity_sign
from .schemas import CanonicalNode, EdgeRecord, StanceRecord


def aggregate_edges(
    edges: list[EdgeRecord],
    registry: dict[str, CanonicalNode],
    filter_fn: Callable[[EdgeRecord], bool] | None = None,
) -> dict[str, Any]:
    """Aggregate concept->concept edges applying an optional filter function.

    Returns nodes (all registry nodes with weights) and aggregated edges with
    mention counts, stance/polarity distributions, and mean signed polarity.
    """
    filtered_edges = [e for e in edges if filter_fn(e)] if filter_fn else edges

    node_weights: dict[str, int] = {nid: 0 for nid in registry}
    for edge in filtered_edges:
        if edge.source_node_id:
            node_weights[edge.source_node_id] = node_weights.get(edge.source_node_id, 0) + 1
        if edge.target_node_id:
            node_weights[edge.target_node_id] = node_weights.get(edge.target_node_id, 0) + 1

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
        polarity_counts: dict[str, int] = defaultdict(int)
        confidences: list[float] = []
        signs: list[int] = []

        for e in group:
            stance_counts[e.stance.value] += 1
            explicitness_counts[e.explicitness.value] += 1
            polarity_counts[e.polarity.value] += 1
            confidences.append(e.confidence)
            signs.append(polarity_sign(e.polarity))

        total = len(group)
        aggregated_edges.append({
            "source_node_id": src,
            "target_node_id": tgt,
            "relation": rel,
            "mention_count": total,
            "unique_speakers": len(speakers),
            "unique_tables": len(tables),
            "unique_rounds": len(rounds),
            "stance_distribution": {k: v / total for k, v in stance_counts.items()},
            "explicitness_distribution": {k: v / total for k, v in explicitness_counts.items()},
            "polarity_distribution": {k: v / total for k, v in polarity_counts.items()},
            "mean_polarity": sum(signs) / len(signs) if signs else 0.0,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "evidence_samples": [e.evidence_text for e in group[:5]],
            "speaker_ids": list(speakers),
        })

    nodes = [
        {
            "node_id": nid,
            "label": node.label,
            "type": node.type.value,
            "aliases": node.aliases,
            "weight": node_weights.get(nid, 0),
        }
        for nid, node in registry.items()
    ]

    return {
        "nodes": nodes,
        "edges": aggregated_edges,
        "total_edges_raw": len(filtered_edges),
        "total_nodes": len(registry),
    }


def aggregate_stances(
    stances: list[StanceRecord],
    filter_fn: Callable[[StanceRecord], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate participant->concept stances into per-concept polarity summaries."""
    filtered = [s for s in stances if filter_fn(s)] if filter_fn else stances

    by_concept: dict[str, list[StanceRecord]] = defaultdict(list)
    for s in filtered:
        if s.concept_node_id:
            by_concept[s.concept_node_id].append(s)

    out: dict[str, dict[str, Any]] = {}
    for cid, group in by_concept.items():
        counts: dict[str, int] = defaultdict(int)
        signs: list[int] = []
        for s in group:
            counts[s.polarity.value] += 1
            signs.append(polarity_sign(s.polarity))
        total = len(group)
        out[cid] = {
            "concept_node_id": cid,
            "stance_count": total,
            "unique_speakers": len(set(s.participant_id for s in group)),
            "polarity_distribution": {k: v / total for k, v in counts.items()},
            "mean_polarity": sum(signs) / len(signs) if signs else 0.0,
        }
    return out


def _affiliation_of(participant_id: str, demographics: dict[str, dict[str, str]], attribute: str) -> str:
    return demographics.get(participant_id, {}).get(attribute, "Unknown")


def compute_group_divergence(
    edges: list[EdgeRecord],
    stances: list[StanceRecord],
    demographics: dict[str, dict[str, str]],
    attribute: str = "political_affiliation",
    group_a: str | None = None,
    group_b: str | None = None,
    top_n: int = 40,
) -> dict[str, Any]:
    """Compare two demographic groups and surface where their causal maps diverge.

    Divergence is computed two ways:
      - concept polarity: |mean_polarity_a - mean_polarity_b| over stances+edges
      - edge presence: a relationship asserted by one group but not the other
    """
    groups = sorted({
        _affiliation_of(e.participant_id, demographics, attribute) for e in edges
    } | {
        _affiliation_of(s.participant_id, demographics, attribute) for s in stances
    } - {"Unknown"})

    if group_a is None or group_b is None:
        # Default to the two largest groups.
        if len(groups) < 2:
            return {"available_groups": groups, "concept_divergence": [], "edge_divergence": []}
        group_a, group_b = groups[0], groups[1]

    def _node_signs(gid: str) -> dict[str, list[int]]:
        acc: dict[str, list[int]] = defaultdict(list)
        for s in stances:
            if s.concept_node_id and _affiliation_of(s.participant_id, demographics, attribute) == gid:
                acc[s.concept_node_id].append(polarity_sign(s.polarity))
        for e in edges:
            if _affiliation_of(e.participant_id, demographics, attribute) != gid:
                continue
            sign = polarity_sign(e.polarity)
            for nid in (e.source_node_id, e.target_node_id):
                if nid:
                    acc[nid].append(sign)
        return acc

    signs_a, signs_b = _node_signs(group_a), _node_signs(group_b)

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    concept_div: list[dict[str, Any]] = []
    for nid in set(signs_a) | set(signs_b):
        ma, mb = _mean(signs_a.get(nid, [])), _mean(signs_b.get(nid, []))
        concept_div.append({
            "concept_node_id": nid,
            "mean_polarity_a": round(ma, 3),
            "mean_polarity_b": round(mb, 3),
            "n_a": len(signs_a.get(nid, [])),
            "n_b": len(signs_b.get(nid, [])),
            "divergence": round(abs(ma - mb), 3),
        })
    concept_div.sort(key=lambda d: (-d["divergence"], -(d["n_a"] + d["n_b"])))

    def _triples(gid: str) -> set[tuple]:
        return {
            (e.source_node_id, e.target_node_id, e.relation.value)
            for e in edges
            if _affiliation_of(e.participant_id, demographics, attribute) == gid
            and e.source_node_id and e.target_node_id
        }

    triples_a, triples_b = _triples(group_a), _triples(group_b)
    only_a = triples_a - triples_b
    only_b = triples_b - triples_a
    shared = triples_a & triples_b

    edge_div = [
        {"source_node_id": s, "target_node_id": t, "relation": r, "only_in": group_a}
        for (s, t, r) in only_a
    ] + [
        {"source_node_id": s, "target_node_id": t, "relation": r, "only_in": group_b}
        for (s, t, r) in only_b
    ]

    union = triples_a | triples_b
    jaccard = len(shared) / len(union) if union else 0.0

    return {
        "attribute": attribute,
        "group_a": group_a,
        "group_b": group_b,
        "available_groups": groups,
        "shared_triples": len(shared),
        "only_a_triples": len(only_a),
        "only_b_triples": len(only_b),
        "triple_jaccard": round(jaccard, 4),
        "concept_divergence": concept_div[:top_n],
        "edge_divergence": edge_div[:top_n],
    }
