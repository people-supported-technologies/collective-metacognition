#!/usr/bin/env python
"""Post-process already-extracted edges/stances: mine/ground vocabulary, export.

Use this to iterate on grounding + visualisation without re-running LLM extraction.

Usage:
    python scripts/post_process.py                 # mine vocab if missing, then ground
    python scripts/post_process.py --remine        # force re-mine the vocabulary
    python scripts/post_process.py --similarity 0.65
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.causalmap.config import PROCESSED_DIR
from src.causalmap.schemas import EdgeRecord, StanceRecord
from src.causalmap.concept_vocab import mine_vocabulary, save_vocab, load_vocab
from src.causalmap.canonicalisation import (
    ground_concepts,
    apply_grounding,
    registry_from_vocab,
    save_grounding_report,
)
from src.causalmap.node_registry import save_registry
from src.causalmap.export import export_graph_data
from src.causalmap.eval import generate_labeling_template, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_edges(path: Path | None = None) -> list[EdgeRecord]:
    path = path or (PROCESSED_DIR / "edges.json")
    with open(path) as f:
        return [EdgeRecord.model_validate(e) for e in json.load(f)]


def load_stances(path: Path | None = None) -> list[StanceRecord]:
    path = path or (PROCESSED_DIR / "stances.json")
    if not path.exists():
        return []
    with open(path) as f:
        return [StanceRecord.model_validate(s) for s in json.load(f)]


def load_turns(path: Path | None = None) -> list[dict]:
    path = path or (PROCESSED_DIR / "canonical_turns.json")
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Post-process extracted edges/stances")
    parser.add_argument("--remine", action="store_true", help="Re-mine the vocabulary even if it exists")
    parser.add_argument("--similarity", type=float, default=0.60, help="Embedding grounding threshold")
    args = parser.parse_args()

    edges = load_edges()
    stances = load_stances()
    turns = load_turns()
    logger.info(f"Loaded {len(edges)} edges, {len(stances)} stances, {len(turns)} turns")

    vocab_path = PROCESSED_DIR / "concept_vocab.json"
    if vocab_path.exists() and not args.remine:
        vocab = load_vocab(vocab_path)
        logger.info(f"Reusing vocabulary: {len(vocab)} concepts")
    else:
        logger.info("Mining controlled vocabulary ...")
        vocab = mine_vocabulary(edges, stances)
        save_vocab(vocab)
        logger.info(f"Mined vocabulary: {len(vocab)} concepts")

    logger.info("Grounding concepts to vocabulary ...")
    vocab, label_to_id, report = ground_concepts(edges, stances, vocab, similarity_threshold=args.similarity)
    edges, stances = apply_grounding(edges, stances, label_to_id)
    save_vocab(vocab)
    save_grounding_report(report)

    registry = registry_from_vocab(vocab)
    reg_path = save_registry(registry)
    logger.info(f"Registry: {len(registry)} concepts -> {reg_path}")

    graph_path = export_graph_data(edges, registry, turns, stances=stances)
    logger.info(f"Graph data -> {graph_path}")

    generate_labeling_template(turns)

    metrics = compute_metrics(edges, stances=stances)
    with open(PROCESSED_DIR / "extraction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Edges: {metrics['total_predicted_edges']} | Stances: {metrics.get('total_stances', 0)}")
    logger.info(f"Nodes (concepts): {len(registry)}")
    logger.info(f"Relations: {metrics['relation_distribution']}")
    logger.info(f"Polarity: {metrics.get('polarity_distribution', {})}")
    logger.info(f"Group overlap: {metrics.get('group_overlap', {})}")
    logger.info("=" * 60)
    logger.info("Run the viewer with: cd app && python -m http.server 8080")


if __name__ == "__main__":
    main()
