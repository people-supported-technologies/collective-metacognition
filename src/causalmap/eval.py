"""Evaluation utilities: labeling template generation and metrics computation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import LABELLED_DIR, PROCESSED_DIR
from .schemas import EdgeRecord


def generate_labeling_template(
    turns: list[dict[str, Any]],
    n_samples: int = 120,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """Generate a JSONL labeling template from sampled turns.

    Each line contains a turn with space for human annotations.
    Samples turns with >5 words to ensure enough content.
    """
    output_dir = output_dir or LABELLED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    substantive = [t for t in turns if len(t["text"].split()) > 5]
    random.seed(seed)
    sample = random.sample(substantive, min(n_samples, len(substantive)))

    out_path = output_dir / "labeling_template.jsonl"
    with open(out_path, "w") as f:
        for turn in sample:
            record = {
                "turn_id": turn["turn_id"],
                "speaker": turn["speaker"],
                "table_id": turn["table_id"],
                "round_id": turn["round_id"],
                "text": turn["text"],
                "annotations": {
                    "concepts": [],
                    "edges": [],
                    "notes": "",
                },
            }
            f.write(json.dumps(record) + "\n")

    return out_path


def compute_metrics(
    predicted_edges: list[EdgeRecord],
    gold_path: Path | None = None,
) -> dict[str, Any]:
    """Compute extraction quality metrics against a gold-standard file.

    If no gold file exists, returns summary stats of predicted edges only.
    """
    stats: dict[str, Any] = {
        "total_predicted_edges": len(predicted_edges),
        "relation_distribution": {},
        "stance_distribution": {},
        "explicitness_distribution": {},
        "confidence_stats": {},
        "causal_vs_value_share": {},
    }

    if not predicted_edges:
        return stats

    from collections import Counter
    import statistics

    rel_dist = Counter(e.relation.value for e in predicted_edges)
    stance_dist = Counter(e.stance.value for e in predicted_edges)
    expl_dist = Counter(e.explicitness.value for e in predicted_edges)
    confidences = [e.confidence for e in predicted_edges]

    stats["relation_distribution"] = dict(rel_dist.most_common())
    stats["stance_distribution"] = dict(stance_dist.most_common())
    stats["explicitness_distribution"] = dict(expl_dist.most_common())
    stats["confidence_stats"] = {
        "mean": statistics.mean(confidences),
        "median": statistics.median(confidences),
        "min": min(confidences),
        "max": max(confidences),
    }

    # Causal relations vs value/stance relations
    causal_rels = {"causes", "increases", "reduces", "enables", "prevents", "reinforces", "undermines", "increases_exposure_to"}
    value_rels = {"expresses", "rejects", "questions", "reports", "supports", "opposes"}

    causal_count = sum(1 for e in predicted_edges if e.relation.value in causal_rels)
    value_count = sum(1 for e in predicted_edges if e.relation.value in value_rels)
    other_count = len(predicted_edges) - causal_count - value_count

    stats["causal_vs_value_share"] = {
        "causal": causal_count,
        "value_stance": value_count,
        "other": other_count,
        "causal_fraction": causal_count / len(predicted_edges),
        "value_fraction": value_count / len(predicted_edges),
    }

    # If gold file exists, compute precision/recall
    if gold_path and gold_path.exists():
        stats["gold_comparison"] = _compare_to_gold(predicted_edges, gold_path)

    return stats


def _compare_to_gold(predicted: list[EdgeRecord], gold_path: Path) -> dict[str, Any]:
    """Compare predicted edges against gold-standard annotations."""
    with open(gold_path) as f:
        gold_data = [json.loads(line) for line in f if line.strip()]

    gold_edges = []
    for item in gold_data:
        for edge in item.get("annotations", {}).get("edges", []):
            gold_edges.append((
                edge.get("source", "").lower(),
                edge.get("relation", "").lower(),
                edge.get("target", "").lower(),
            ))

    if not gold_edges:
        return {"note": "No gold edges found"}

    pred_triples = set()
    for e in predicted:
        pred_triples.add((
            e.source_node.label.lower(),
            e.relation.value,
            e.target_node.label.lower(),
        ))

    gold_set = set(gold_edges)
    tp = len(pred_triples & gold_set)
    precision = tp / len(pred_triples) if pred_triples else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "predicted_count": len(pred_triples),
        "gold_count": len(gold_set),
    }
