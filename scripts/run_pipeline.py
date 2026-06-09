#!/usr/bin/env python
"""CLI to run the full grounded extraction pipeline.

Flow:
  preprocess -> acquire controlled vocabulary -> grounded extraction
  -> ground concepts to vocab -> build registry -> export -> eval

Vocabulary acquisition (in priority order):
  1. --remine off and concept_vocab.json exists  -> reuse it
  2. existing edges.json present (not --bootstrap) -> mine vocab from it
  3. otherwise                                     -> bootstrap extraction, then mine

Usage:
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --tables 3 --bootstrap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.causalmap.config import PROCESSED_DIR
from src.causalmap.loader import load_discussion_segments
from src.causalmap.preprocess import (
    pseudonymize_turns,
    reconstruct_utterances,
    write_canonical_turns,
)
from src.causalmap.extract import run_extraction
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
from src.causalmap.schemas import EdgeRecord, StanceRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_busiest_tables(turns: list[dict], n: int = 5) -> list[str]:
    counter = Counter(t["table_id"] for t in turns if len(t["text"].split()) > 3)
    return [table_id for table_id, _ in counter.most_common(n)]


def _load_edges(path: Path) -> list[EdgeRecord]:
    with open(path) as f:
        return [EdgeRecord.model_validate(e) for e in json.load(f)]


def _load_stances(path: Path) -> list[StanceRecord]:
    if not path.exists():
        return []
    with open(path) as f:
        return [StanceRecord.model_validate(s) for s in json.load(f)]


def acquire_vocabulary(turns, table_ids, args):
    vocab_path = PROCESSED_DIR / "concept_vocab.json"
    edges_path = PROCESSED_DIR / "edges.json"

    if vocab_path.exists() and not args.remine:
        logger.info(f"Reusing existing vocabulary at {vocab_path}")
        return load_vocab(vocab_path)

    if edges_path.exists() and not args.bootstrap:
        logger.info("Mining vocabulary from existing edges.json ...")
        seed_edges = _load_edges(edges_path)
        seed_stances = _load_stances(PROCESSED_DIR / "stances.json")
    else:
        logger.info("Bootstrap extraction (no vocabulary) to gather concept candidates ...")
        seed_edges, seed_stances = run_extraction(turns, vocab=None, table_ids=table_ids, max_workers=args.workers)

    vocab = mine_vocabulary(seed_edges, seed_stances)
    save_vocab(vocab)
    logger.info(f"Controlled vocabulary: {len(vocab)} concepts")
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Run grounded causal map extraction pipeline")
    parser.add_argument("--tables", type=int, default=3, help="Number of busiest tables to process")
    parser.add_argument("--all", action="store_true", help="Process all tables")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap-extract to mine vocab instead of reusing edges.json")
    parser.add_argument("--remine", action="store_true", help="Re-mine the vocabulary even if concept_vocab.json exists")
    args = parser.parse_args()

    logger.info("Loading segments from transcription.xlsx ...")
    segments = load_discussion_segments()
    turns = reconstruct_utterances(segments)
    turns = pseudonymize_turns(turns)
    write_canonical_turns(turns)
    logger.info(f"{len(turns)} canonical turns")

    table_ids = None if args.all else get_busiest_tables(turns, n=args.tables)
    logger.info("Processing ALL tables" if args.all else f"Processing tables: {table_ids}")

    vocab = acquire_vocabulary(turns, table_ids, args)

    logger.info("Grounded extraction ...")
    edges, stances = run_extraction(turns, vocab=vocab, table_ids=table_ids, max_workers=args.workers)
    logger.info(f"{len(edges)} edges, {len(stances)} stances")

    logger.info("Grounding concepts to vocabulary ...")
    vocab, label_to_id, report = ground_concepts(edges, stances, vocab)
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
    logger.info(f"Relations: {metrics['relation_distribution']}")
    logger.info(f"Polarity: {metrics.get('polarity_distribution', {})}")
    logger.info(f"Group overlap: {metrics.get('group_overlap', {})}")
    logger.info("=" * 60)
    logger.info("Run the viewer with: cd app && python -m http.server 8080")


if __name__ == "__main__":
    main()
