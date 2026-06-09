"""Export graph_data.json for the D3 viewer.

Produces the full node list (always present), concept->concept edges with signed
polarity, the participant->concept stance overlay, and a precomputed Dem-vs-Rep
divergence summary so the viewer can highlight where group maps differ.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .aggregate import aggregate_edges, aggregate_stances, compute_group_divergence
from .config import PROCESSED_DIR
from .demographics import load_normalized_demographics
from .ontology import polarity_sign
from .schemas import CanonicalNode, EdgeRecord, StanceRecord


def export_graph_data(
    edges: list[EdgeRecord],
    registry: dict[str, CanonicalNode],
    turns: list[dict[str, Any]],
    stances: list[StanceRecord] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Export the full graph data JSON for the D3 viewer."""
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stances = stances or []

    agg = aggregate_edges(edges, registry)
    stance_summary = aggregate_stances(stances)

    speakers = sorted(set(e.participant_id for e in edges) | set(s.participant_id for s in stances))
    tables = sorted(set(e.table_id for e in edges))
    rounds = sorted(set(e.round_id for e in edges))

    all_demographics = load_normalized_demographics()
    participant_ids = set(speakers)
    participant_demographics = {pid: all_demographics.get(pid, {}) for pid in participant_ids}

    edge_affiliation_counts = Counter(
        participant_demographics[e.participant_id].get("political_affiliation", "Unknown")
        for e in edges
    )
    speaker_affiliation_counts = Counter(
        participant_demographics[pid].get("political_affiliation", "Unknown")
        for pid in participant_ids
    )
    demographic_filters = {
        "political_affiliation": [
            {
                "value": "all",
                "label": "All participants",
                "edge_count": len(edges),
                "speaker_count": len(participant_ids),
            },
            *[
                {
                    "value": value,
                    "label": value,
                    "edge_count": edge_affiliation_counts.get(value, 0),
                    "speaker_count": speaker_affiliation_counts.get(value, 0),
                }
                for value in sorted(
                    (v for v in edge_affiliation_counts if v != "Unknown"),
                    key=lambda v: (-edge_affiliation_counts[v], v),
                )
            ],
        ],
    }

    raw_edges = [
        {
            "edge_id": edge.edge_id,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            "relation": edge.relation.value,
            "speaker": edge.speaker,
            "participant_id": edge.participant_id,
            "table_id": edge.table_id,
            "round_id": edge.round_id,
            "evidence_text": edge.evidence_text,
            "stance": edge.stance.value,
            "polarity": edge.polarity.value,
            "polarity_sign": polarity_sign(edge.polarity),
            "explicitness": edge.explicitness.value,
            "confidence": edge.confidence,
        }
        for edge in edges
    ]

    raw_stances = [
        {
            "stance_id": s.stance_id,
            "concept_node_id": s.concept_node_id,
            "concept_label": s.concept.label,
            "concept_type": s.concept.type.value,
            "polarity": s.polarity.value,
            "polarity_sign": polarity_sign(s.polarity),
            "speaker": s.speaker,
            "participant_id": s.participant_id,
            "table_id": s.table_id,
            "round_id": s.round_id,
            "evidence_text": s.evidence_text,
            "explicitness": s.explicitness.value,
            "confidence": s.confidence,
        }
        for s in stances
    ]

    divergence = compute_group_divergence(edges, stances, all_demographics, attribute="political_affiliation")

    graph_data = {
        "nodes": agg["nodes"],
        "aggregated_edges": agg["edges"],
        "raw_edges": raw_edges,
        "raw_stances": raw_stances,
        "stance_summary": stance_summary,
        "participant_demographics": participant_demographics,
        "group_divergence": divergence,
        "metadata": {
            "speakers": speakers,
            "tables": tables,
            "rounds": rounds,
            "demographic_filters": demographic_filters,
            "total_raw_edges": len(edges),
            "total_raw_stances": len(stances),
            "total_nodes": len(registry),
        },
    }

    out_path = output_dir / "graph_data.json"
    with open(out_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    app_dir = output_dir.parent.parent / "app"
    if app_dir.exists():
        import shutil
        shutil.copy2(out_path, app_dir / "graph_data.json")

    return out_path
