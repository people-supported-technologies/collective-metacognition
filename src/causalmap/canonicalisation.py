"""Vocabulary grounding: snap every extracted concept onto a controlled-vocab entry.

This replaces the old agglomerative post-hoc merge of sentence-like labels (which
could only move shared triples 11 -> 12). With grounded extraction, most concepts
already use the canonical vocabulary labels; this module is the reconciliation /
safety net that:
  - maps exact label/alias matches to their concept_id,
  - snaps remaining labels to their nearest vocabulary concept by embedding
    similarity (same type, above threshold),
  - otherwise registers a new vocabulary concept,
and emits an auditable grounding report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import PROCESSED_DIR
from .schemas import ConceptVocabEntry, EdgeRecord, StanceRecord

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY = 0.60  # cosine; snap a mention to a vocab concept above this


def _normalize(label: str) -> str:
    return re.sub(r"\s+", " ", label.lower().strip())


def _concept_id(label: str) -> str:
    h = hashlib.sha256(_normalize(label).encode()).hexdigest()[:12]
    return f"concept_{h}"


def _collect_mentions(
    edges: list[EdgeRecord],
    stances: list[StanceRecord],
) -> dict[str, dict[str, Any]]:
    """Unique mention labels -> {label, type, count}."""
    mentions: dict[str, dict[str, Any]] = {}

    def _add(label: str, type_value: str) -> None:
        norm = _normalize(label)
        if not norm:
            return
        if norm not in mentions:
            mentions[norm] = {"label": label.strip(), "type": type_value, "count": 0}
        mentions[norm]["count"] += 1

    for e in edges:
        _add(e.source_node.label, e.source_node.type.value)
        _add(e.target_node.label, e.target_node.type.value)
    for s in stances:
        _add(s.concept.label, s.concept.type.value)
    return mentions


def ground_concepts(
    edges: list[EdgeRecord],
    stances: list[StanceRecord],
    vocab: list[ConceptVocabEntry],
    similarity_threshold: float = DEFAULT_SIMILARITY,
    model_name: str = DEFAULT_MODEL,
) -> tuple[list[ConceptVocabEntry], dict[str, str], list[dict[str, Any]]]:
    """Resolve every extracted mention to a vocabulary concept_id.

    Returns:
        updated_vocab: vocab plus any newly registered concepts / aliases
        label_to_id:   normalized mention label -> concept_id
        report:        per-mention grounding decision (for human review)
    """
    mentions = _collect_mentions(edges, stances)

    # Index vocab by exact label and by alias for cheap matches.
    by_label: dict[str, ConceptVocabEntry] = {}
    by_alias: dict[str, ConceptVocabEntry] = {}
    vocab_index: dict[str, ConceptVocabEntry] = {v.concept_id: v for v in vocab}
    for v in vocab:
        by_label[_normalize(v.label)] = v
        for alias in v.aliases:
            by_alias.setdefault(_normalize(alias), v)

    label_to_id: dict[str, str] = {}
    report: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for norm, info in mentions.items():
        if norm in by_label:
            label_to_id[norm] = by_label[norm].concept_id
            report.append({"label": info["label"], "method": "exact_label", "concept_id": by_label[norm].concept_id, "concept_label": by_label[norm].label})
        elif norm in by_alias:
            label_to_id[norm] = by_alias[norm].concept_id
            report.append({"label": info["label"], "method": "alias", "concept_id": by_alias[norm].concept_id, "concept_label": by_alias[norm].label})
        else:
            unmatched.append(norm)

    if unmatched and vocab:
        logger.info(f"Embedding-grounding {len(unmatched)} unmatched mentions against {len(vocab)} concepts")
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        vocab_labels = [v.label for v in vocab]
        vocab_types = [v.type.value for v in vocab]
        vocab_emb = model.encode(vocab_labels, normalize_embeddings=True, show_progress_bar=False)
        mention_emb = model.encode([mentions[n]["label"] for n in unmatched], normalize_embeddings=True, show_progress_bar=False)

        sims = np.asarray(mention_emb) @ np.asarray(vocab_emb).T
        for i, norm in enumerate(unmatched):
            m_type = mentions[norm]["type"]
            # Prefer same-type matches; mask out other types unless none qualify.
            row = sims[i].copy()
            same_type_mask = np.array([t == m_type for t in vocab_types])
            masked = np.where(same_type_mask, row, -1.0)
            best_idx = int(masked.argmax()) if same_type_mask.any() else int(row.argmax())
            best_sim = float(row[best_idx])
            if best_sim >= similarity_threshold:
                target = vocab[best_idx]
                label_to_id[norm] = target.concept_id
                if _normalize(mentions[norm]["label"]) not in {_normalize(a) for a in target.aliases}:
                    target.aliases.append(mentions[norm]["label"])
                report.append({"label": mentions[norm]["label"], "method": "embedding", "similarity": round(best_sim, 3), "concept_id": target.concept_id, "concept_label": target.label})
            else:
                unmatched_new = _register_new(mentions[norm], vocab_index, by_label)
                label_to_id[norm] = unmatched_new.concept_id
                report.append({"label": mentions[norm]["label"], "method": "new_concept", "similarity": round(best_sim, 3), "concept_id": unmatched_new.concept_id, "concept_label": unmatched_new.label})
    else:
        for norm in unmatched:
            new_entry = _register_new(mentions[norm], vocab_index, by_label)
            label_to_id[norm] = new_entry.concept_id
            report.append({"label": mentions[norm]["label"], "method": "new_concept", "concept_id": new_entry.concept_id, "concept_label": new_entry.label})

    updated_vocab = list(vocab_index.values())
    methods = defaultdict(int)
    for r in report:
        methods[r["method"]] += 1
    logger.info(f"Grounded {len(mentions)} mentions: {dict(methods)}; vocab now {len(updated_vocab)} concepts")
    return updated_vocab, label_to_id, report


def _register_new(
    info: dict[str, Any],
    vocab_index: dict[str, ConceptVocabEntry],
    by_label: dict[str, ConceptVocabEntry],
) -> ConceptVocabEntry:
    from .ontology import NodeType

    norm = _normalize(info["label"])
    cid = _concept_id(info["label"])
    if cid in vocab_index:
        return vocab_index[cid]
    entry = ConceptVocabEntry(
        concept_id=cid,
        label=norm,
        type=NodeType(info["type"]),
        aliases=[info["label"]],
        mention_count=info["count"],
    )
    vocab_index[cid] = entry
    by_label[norm] = entry
    return entry


def apply_grounding(
    edges: list[EdgeRecord],
    stances: list[StanceRecord],
    label_to_id: dict[str, str],
) -> tuple[list[EdgeRecord], list[StanceRecord]]:
    """Populate concept ids on edges (source/target) and stance records."""
    for e in edges:
        e.source_node_id = label_to_id.get(_normalize(e.source_node.label))
        e.target_node_id = label_to_id.get(_normalize(e.target_node.label))
        e.source_node.concept_id = e.source_node_id
        e.target_node.concept_id = e.target_node_id
    for s in stances:
        s.concept_node_id = label_to_id.get(_normalize(s.concept.label))
        s.concept.concept_id = s.concept_node_id
    return edges, stances


def registry_from_vocab(vocab: list[ConceptVocabEntry]) -> dict[str, Any]:
    """Build the fixed node registry (concept_id -> CanonicalNode) from the vocabulary."""
    from .schemas import CanonicalNode

    registry: dict[str, CanonicalNode] = {}
    for v in vocab:
        registry[v.concept_id] = CanonicalNode(
            node_id=v.concept_id,
            label=v.label,
            type=v.type,
            aliases=v.aliases,
        )
    return registry


def save_grounding_report(report: list[dict[str, Any]], output_dir: Path | None = None) -> Path:
    """Persist the grounding report for human review."""
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "grounding_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path
