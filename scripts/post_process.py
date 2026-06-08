#!/usr/bin/env python
"""Post-process already-extracted edges: build registry, aggregate, export graph_data.json.

Usage:
    python scripts/post_process.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.causalmap.config import PROCESSED_DIR
from src.causalmap.schemas import EdgeRecord
from src.causalmap.node_registry import build_registry, assign_node_ids, save_registry
from src.causalmap.export import export_graph_data
from src.causalmap.eval import generate_labeling_template, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_edges(path: Path | None = None) -> list[EdgeRecord]:
    """Load edges from the extraction output JSON."""
    path = path or (PROCESSED_DIR / "edges.json")
    with open(path) as f:
        data = json.load(f)
    return [EdgeRecord.model_validate(e) for e in data]


def load_turns(path: Path | None = None) -> list[dict]:
    """Load canonical turns."""
    path = path or (PROCESSED_DIR / "canonical_turns.json")
    with open(path) as f:
        return json.load(f)


def main():
    logger.info("Loading edges...")
    edges = load_edges()
    logger.info(f"Loaded {len(edges)} edges")

    logger.info("Loading turns...")
    turns = load_turns()
    logger.info(f"Loaded {len(turns)} turns")

    logger.info("Building node registry...")
    registry = build_registry(edges)
    edges = assign_node_ids(edges, registry)
    reg_path = save_registry(registry)
    logger.info(f"Node registry: {len(registry)} canonical nodes -> {reg_path}")

    logger.info("Exporting graph data for viewer...")
    graph_path = export_graph_data(edges, registry, turns)
    logger.info(f"Graph data -> {graph_path}")

    logger.info("Generating labeling template...")
    label_path = generate_labeling_template(turns)
    logger.info(f"Labeling template -> {label_path}")

    logger.info("Computing metrics...")
    metrics = compute_metrics(edges)
    metrics_path = PROCESSED_DIR / "extraction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics -> {metrics_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info(f"Total edges: {metrics['total_predicted_edges']}")
    logger.info(f"Relations: {metrics['relation_distribution']}")
    logger.info(f"Stances: {metrics['stance_distribution']}")
    logger.info(f"Explicitness: {metrics['explicitness_distribution']}")
    logger.info(f"Confidence: {metrics['confidence_stats']}")
    logger.info(f"Causal vs value: {metrics['causal_vs_value_share']}")
    logger.info("=" * 60)
    logger.info("Done. Run the viewer with: cd app && python -m http.server 8080")


if __name__ == "__main__":
    main()
