"""Export graph_data.json for the D3 viewer.

Produces the full node list (always present) with per-edge provenance and
weight information so the viewer can recompute display weights client-side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from collections import Counter

from .config import PROCESSED_DIR
from .schemas import CanonicalNode, EdgeRecord
from .aggregate import aggregate_edges
from .demographics import load_normalized_demographics


def export_graph_data(
    edges: list[EdgeRecord],
    registry: dict[str, CanonicalNode],
    turns: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> Path:
    """Export the full graph data JSON for the D3 viewer.

    The JSON contains:
      - nodes: full registry (always all nodes)
      - edges: raw edges with full attribution for client-side filtering
      - metadata: filter options (speakers, tables, rounds)
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build aggregated view (unfiltered = full graph)
    agg = aggregate_edges(edges, registry)

    # Collect filter metadata
    speakers = sorted(set(e.participant_id for e in edges))
    tables = sorted(set(e.table_id for e in edges))
    rounds = sorted(set(e.round_id for e in edges))

    all_demographics = load_normalized_demographics()
    participant_ids = {e.participant_id for e in edges}
    participant_demographics = {
        pid: all_demographics.get(pid, {})
        for pid in participant_ids
    }

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

    # Build raw edges for client-side filtering
    raw_edges = []
    for edge in edges:
        raw_edges.append({
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
            "explicitness": edge.explicitness.value,
            "confidence": edge.confidence,
        })

    graph_data = {
        "nodes": agg["nodes"],
        "aggregated_edges": agg["edges"],
        "raw_edges": raw_edges,
        "participant_demographics": participant_demographics,
        "metadata": {
            "speakers": speakers,
            "tables": tables,
            "rounds": rounds,
            "demographic_filters": demographic_filters,
            "total_raw_edges": len(edges),
            "total_nodes": len(registry),
        },
    }

    out_path = output_dir / "graph_data.json"
    with open(out_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    # Also copy to app/ for the D3 viewer
    app_dir = output_dir.parent.parent / "app"
    if app_dir.exists():
        import shutil
        shutil.copy2(out_path, app_dir / "graph_data.json")

    return out_path
