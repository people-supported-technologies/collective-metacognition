#!/usr/bin/env python
"""CLI to run the full extraction pipeline.

Usage:
    python scripts/run_pipeline.py [--tables N] [--all]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.causalmap.loader import load_discussion_segments
from src.causalmap.preprocess import (
    pseudonymize_turns,
    reconstruct_utterances,
    write_canonical_turns,
)
from src.causalmap.extract import run_extraction
from src.causalmap.node_registry import build_registry, assign_node_ids, save_registry
from src.causalmap.export import export_graph_data
from src.causalmap.eval import generate_labeling_template, compute_metrics
from src.causalmap.config import PROCESSED_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_busiest_tables(turns: list[dict], n: int = 5) -> list[str]:
    """Return the N table_ids with the most turns (excluding very short tables)."""
    counter = Counter(t["table_id"] for t in turns if len(t["text"].split()) > 3)
    return [table_id for table_id, _ in counter.most_common(n)]


def main():
    parser = argparse.ArgumentParser(description="Run causal map extraction pipeline")
    parser.add_argument("--tables", type=int, default=3, help="Number of busiest tables to process")
    parser.add_argument("--all", action="store_true", help="Process all tables")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: cpu_count * 4, max 32)")
    args = parser.parse_args()

    logger.info("Loading segments from transcription.xlsx...")
    segments = load_discussion_segments()
    logger.info(f"Loaded {len(segments)} segments")

    logger.info("Reconstructing utterances...")
    turns = reconstruct_utterances(segments)
    logger.info(f"Reconstructed {len(turns)} turns")

    logger.info("Pseudonymizing speakers...")
    turns = pseudonymize_turns(turns)

    logger.info("Writing canonical turns...")
    path = write_canonical_turns(turns)
    logger.info(f"Canonical turns written to {path}")

    if args.all:
        table_ids = None
        logger.info("Processing ALL tables")
    else:
        table_ids = get_busiest_tables(turns, n=args.tables)
        logger.info(f"Processing {len(table_ids)} busiest tables: {table_ids}")

    logger.info("Starting extraction...")
    edges = run_extraction(turns, table_ids=table_ids, max_workers=args.workers)
    logger.info(f"Pipeline complete. {len(edges)} edges extracted.")

    # Build node registry
    logger.info("Building node registry...")
    registry = build_registry(edges)
    edges = assign_node_ids(edges, registry)
    reg_path = save_registry(registry)
    logger.info(f"Node registry: {len(registry)} nodes, saved to {reg_path}")

    # Export graph data for viewer
    logger.info("Exporting graph data...")
    graph_path = export_graph_data(edges, registry, turns)
    logger.info(f"Graph data exported to {graph_path}")

    # Generate labeling template
    logger.info("Generating labeling template...")
    label_path = generate_labeling_template(turns)
    logger.info(f"Labeling template written to {label_path}")

    # Compute metrics
    metrics = compute_metrics(edges)
    metrics_path = PROCESSED_DIR / "extraction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics written to {metrics_path}")

    # Quick stats
    if edges:
        from collections import Counter
        rel_dist = Counter(e.relation.value for e in edges)
        stance_dist = Counter(e.stance.value for e in edges)
        expl_dist = Counter(e.explicitness.value for e in edges)
        logger.info(f"Relations: {dict(rel_dist)}")
        logger.info(f"Stances: {dict(stance_dist)}")
        logger.info(f"Explicitness: {dict(expl_dist)}")
        logger.info(f"Causal vs value: {metrics.get('causal_vs_value_share', {})}")


if __name__ == "__main__":
    main()
