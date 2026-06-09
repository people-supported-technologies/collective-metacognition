"""Controlled concept vocabulary mining (the shared backbone for group comparison).

The core failure of the previous design was that every speaker's phrasing became
its own node, so groups had almost nothing to differ over. This module builds a
compact, dataset-wide controlled vocabulary of short, polarity-NEUTRAL concepts
that re-extraction then grounds every mention into.

Pipeline:
  1. Collect candidate concept mentions (from existing edges + stances).
  2. Embedding-cluster near-duplicate phrasings into proto-concepts.
  3. LLM consolidation: name each proto-concept with a short neutral label + type,
     reusing identical labels for synonymous clusters.
  4. Tight embedding merge of the LLM labels to collapse residual duplicates.
  5. Emit concept_vocab.json (concept_id, label, type, aliases, mention_count).

Speaker pseudonyms are never concepts and are dropped here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import PROCESSED_DIR
from .llm_client import generate_structured
from .ontology import NodeType
from .schemas import ConceptVocabEntry, EdgeRecord, StanceRecord

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
# Candidate clustering: relatively loose so near-duplicate phrasings group together.
DEFAULT_CLUSTER_SIMILARITY = 0.55
# Final merge of LLM-produced canonical labels: tight, to only collapse true duplicates.
DEFAULT_LABEL_MERGE_SIMILARITY = 0.88
CONSOLIDATION_BATCH = 40
REPRESENTATIVES_PER_CLUSTER = 6

_SPEAKER_PREFIX = "speaker_"


def _normalize(label: str) -> str:
    return re.sub(r"\s+", " ", label.lower().strip())


def _concept_id(canonical_label: str) -> str:
    h = hashlib.sha256(_normalize(canonical_label).encode()).hexdigest()[:12]
    return f"concept_{h}"


def _is_speaker_label(label: str) -> bool:
    return label.lower().startswith(_SPEAKER_PREFIX)


# --------------------------------------------------------------------------- #
# 1. Candidate collection
# --------------------------------------------------------------------------- #

def collect_candidates(
    edges: list[EdgeRecord],
    stances: list[StanceRecord] | None = None,
) -> list[dict[str, Any]]:
    """Gather unique candidate concept labels with majority type and mention count."""
    type_votes: dict[str, Counter] = defaultdict(Counter)
    counts: Counter = Counter()
    original: dict[str, str] = {}

    def _add(label: str, type_value: str) -> None:
        if not label or _is_speaker_label(label):
            return
        norm = _normalize(label)
        if not norm:
            return
        counts[norm] += 1
        type_votes[norm][type_value] += 1
        original.setdefault(norm, label.strip())

    for e in edges:
        _add(e.source_node.label, e.source_node.type.value)
        _add(e.target_node.label, e.target_node.type.value)
    for s in stances or []:
        _add(s.concept.label, s.concept.type.value)

    candidates: list[dict[str, Any]] = []
    for norm, count in counts.items():
        majority_type = type_votes[norm].most_common(1)[0][0]
        candidates.append({
            "label": original[norm],
            "norm": norm,
            "type": majority_type,
            "count": count,
        })
    candidates.sort(key=lambda c: (-c["count"], c["norm"]))
    logger.info(f"Collected {len(candidates)} unique candidate concepts")
    return candidates


# --------------------------------------------------------------------------- #
# 2. Embedding clustering
# --------------------------------------------------------------------------- #

def _embed(labels: list[str], model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(labels, normalize_embeddings=True, show_progress_bar=False)


def cluster_candidates(
    candidates: list[dict[str, Any]],
    similarity: float = DEFAULT_CLUSTER_SIMILARITY,
    model_name: str = DEFAULT_MODEL,
) -> list[list[dict[str, Any]]]:
    """Cluster near-duplicate candidate labels into proto-concepts (within type)."""
    from sklearn.cluster import AgglomerativeClustering

    clusters: list[list[dict[str, Any]]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_type[c["type"]].append(c)

    for type_value, items in by_type.items():
        if len(items) == 1:
            clusters.append(items)
            continue
        embeddings = _embed([c["label"] for c in items], model_name)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - similarity,
        )
        labels = clustering.fit_predict(embeddings)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item, cid in zip(items, labels):
            grouped[cid].append(item)
        clusters.extend(grouped.values())

    logger.info(f"Formed {len(clusters)} proto-concept clusters")
    return clusters


# --------------------------------------------------------------------------- #
# 3. LLM consolidation
# --------------------------------------------------------------------------- #

class _VocabAssignment(BaseModel):
    cluster_index: int
    canonical_label: str
    type: NodeType


class _VocabBatch(BaseModel):
    assignments: list[_VocabAssignment]


_CONSOLIDATION_SYSTEM = """You normalise messy conversational concept phrasings into a clean, shared concept vocabulary.

You are given numbered clusters of phrases that participants used in a political discussion. For each cluster, produce ONE canonical concept label following these rules:
1. Keep it SHORT: a 2-4 word noun phrase (e.g. "abortion legality", "border security", "immigration enforcement", "gun control").
2. POLARITY-NEUTRAL: never encode a stance or position. Use "abortion legality" NOT "abortion should be legal"; use "gun control" NOT "ban guns" / "no gun control". Opposite opinions must map to the SAME neutral concept.
3. No negations, no verbs of belief, no "should", no speaker references.
4. Use the SAME canonical_label for two clusters if they refer to the same underlying concept (consolidate synonyms).
5. Pick the most fitting type from the allowed list.

Allowed types: actor, technology, policy, event, process, outcome, value, belief_proposition, topic"""


def _consolidate_batch(batch: list[tuple[int, list[dict[str, Any]]]]) -> list[_VocabAssignment]:
    lines = ["Normalise these concept clusters. Return JSON {\"assignments\": [{\"cluster_index\", \"canonical_label\", \"type\"}]}.\n"]
    for idx, members in batch:
        reps = sorted(members, key=lambda m: -m["count"])[:REPRESENTATIVES_PER_CLUSTER]
        phrases = "; ".join(m["label"] for m in reps)
        lines.append(f"Cluster {idx} ({members[0]['type']}): {phrases}")
    prompt = "\n".join(lines)

    try:
        result = generate_structured(
            prompt=prompt,
            schema=_VocabBatch,
            system_prompt=_CONSOLIDATION_SYSTEM,
            temperature=0.0,
        )
        return result.assignments
    except Exception as e:  # noqa: BLE001
        logger.error(f"Consolidation batch failed: {e}; falling back to representative labels")
        # Fallback: use the most-mentioned member label, lightly trimmed.
        out: list[_VocabAssignment] = []
        for idx, members in batch:
            rep = max(members, key=lambda m: m["count"])
            out.append(_VocabAssignment(
                cluster_index=idx,
                canonical_label=rep["label"],
                type=NodeType(rep["type"]),
            ))
        return out


def consolidate_clusters(
    clusters: list[list[dict[str, Any]]],
    batch_size: int = CONSOLIDATION_BATCH,
    max_workers: int | None = None,
) -> dict[int, _VocabAssignment]:
    """Name each cluster with a neutral canonical label via batched LLM calls."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    indexed = list(enumerate(clusters))
    batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]
    logger.info(f"Consolidating {len(clusters)} clusters in {len(batches)} LLM batches")

    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 2)

    assignments: dict[int, _VocabAssignment] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_consolidate_batch, b): b for b in batches}
        done = 0
        for future in as_completed(futures):
            done += 1
            batch = futures[future]
            valid_idx = {idx for idx, _ in batch}
            for a in future.result():
                if a.cluster_index in valid_idx:
                    assignments[a.cluster_index] = a
            if done % 5 == 0 or done == len(batches):
                logger.info(f"Consolidated {done}/{len(batches)} batches")

    # Any cluster the model skipped: fall back to its representative label.
    for idx, members in indexed:
        if idx not in assignments:
            rep = max(members, key=lambda m: m["count"])
            assignments[idx] = _VocabAssignment(
                cluster_index=idx, canonical_label=rep["label"], type=NodeType(rep["type"])
            )
    return assignments


# --------------------------------------------------------------------------- #
# 4. Final label merge + vocab assembly
# --------------------------------------------------------------------------- #

def build_vocab(
    clusters: list[list[dict[str, Any]]],
    assignments: dict[int, _VocabAssignment],
    label_merge_similarity: float = DEFAULT_LABEL_MERGE_SIMILARITY,
    model_name: str = DEFAULT_MODEL,
) -> list[ConceptVocabEntry]:
    """Collapse synonymous canonical labels and assemble vocabulary entries."""
    from sklearn.cluster import AgglomerativeClustering

    # Group clusters by their (normalized) canonical label first.
    by_label: dict[str, dict[str, Any]] = {}
    for idx, members in enumerate(clusters):
        a = assignments[idx]
        norm = _normalize(a.canonical_label)
        if norm not in by_label:
            by_label[norm] = {
                "label": a.canonical_label.strip(),
                "type": a.type.value,
                "aliases": set(),
                "count": 0,
            }
        entry = by_label[norm]
        entry["count"] += sum(m["count"] for m in members)
        for m in members:
            entry["aliases"].add(m["label"])

    norm_labels = list(by_label.keys())

    # Tight embedding merge to fold residual duplicates (same type only).
    final_groups: dict[str, list[str]] = {}
    if len(norm_labels) > 1:
        by_type: dict[str, list[str]] = defaultdict(list)
        for norm in norm_labels:
            by_type[by_label[norm]["type"]].append(norm)
        for type_value, norms in by_type.items():
            if len(norms) == 1:
                final_groups.setdefault(norms[0], []).append(norms[0])
                continue
            embeddings = _embed([by_label[n]["label"] for n in norms], model_name)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=1.0 - label_merge_similarity,
            )
            labels = clustering.fit_predict(embeddings)
            grouped: dict[int, list[str]] = defaultdict(list)
            for n, cid in zip(norms, labels):
                grouped[cid].append(n)
            for members in grouped.values():
                # canonical = most-mentioned label in the merged group
                head = max(members, key=lambda n: by_label[n]["count"])
                final_groups[head] = members
    else:
        for n in norm_labels:
            final_groups[n] = [n]

    vocab: list[ConceptVocabEntry] = []
    for head, members in final_groups.items():
        aliases: set[str] = set()
        total = 0
        for n in members:
            aliases.update(by_label[n]["aliases"])
            total += by_label[n]["count"]
        label = by_label[head]["label"]
        vocab.append(ConceptVocabEntry(
            concept_id=_concept_id(label),
            label=_normalize(label),
            type=NodeType(by_label[head]["type"]),
            aliases=sorted(aliases),
            mention_count=total,
        ))

    # Deduplicate by concept_id (collisions if two heads normalise identically).
    deduped: dict[str, ConceptVocabEntry] = {}
    for entry in vocab:
        if entry.concept_id in deduped:
            existing = deduped[entry.concept_id]
            existing.aliases = sorted(set(existing.aliases) | set(entry.aliases))
            existing.mention_count += entry.mention_count
        else:
            deduped[entry.concept_id] = entry

    result = sorted(deduped.values(), key=lambda v: -v.mention_count)
    logger.info(f"Built controlled vocabulary of {len(result)} concepts")
    return result


def mine_vocabulary(
    edges: list[EdgeRecord],
    stances: list[StanceRecord] | None = None,
    cluster_similarity: float = DEFAULT_CLUSTER_SIMILARITY,
    label_merge_similarity: float = DEFAULT_LABEL_MERGE_SIMILARITY,
    model_name: str = DEFAULT_MODEL,
) -> list[ConceptVocabEntry]:
    """End-to-end vocabulary mining from existing extracted concepts."""
    candidates = collect_candidates(edges, stances)
    clusters = cluster_candidates(candidates, cluster_similarity, model_name)
    assignments = consolidate_clusters(clusters)
    return build_vocab(clusters, assignments, label_merge_similarity, model_name)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_vocab(vocab: list[ConceptVocabEntry], output_dir: Path | None = None) -> Path:
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "concept_vocab.json"
    with open(out_path, "w") as f:
        json.dump([v.model_dump(mode="json") for v in vocab], f, indent=2)
    return out_path


def load_vocab(path: Path | None = None) -> list[ConceptVocabEntry]:
    path = path or (PROCESSED_DIR / "concept_vocab.json")
    with open(path) as f:
        data = json.load(f)
    return [ConceptVocabEntry.model_validate(v) for v in data]
