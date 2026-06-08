"""Extraction pipeline: windowing, prompt assembly, LLM call, validation.

Processes canonical turns in sliding windows, calls the LLM for each window,
validates evidence spans, and produces atomic EdgeRecords.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LLM_MODEL, LLM_PROVIDER, PROCESSED_DIR, WINDOW_OVERLAP, WINDOW_SIZE
from .llm_client import generate_structured, _build_json_schema_instruction
from .ontology import Explicitness, NodeType, RelationType, Stance
from .schemas import EdgeRecord, ExtractionResult, LLMEdge, Node

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at extracting causal and influence relationships from conversation transcripts.

You will be given a short excerpt from a group discussion. Your task is to identify relationships between concepts mentioned by the speakers.

Rules:
1. Only extract relationships that are DIRECTLY SUPPORTED by the text.
2. The evidence_text MUST be a verbatim quote from the transcript excerpt (copy-paste exactly).
3. Use only the allowed relation types and node types.
4. Assign stance: "asserted" if the speaker states it as true, "rejected" if they deny it, "questioned" if they express doubt, "reported" if they attribute it to others, "uncertain" if ambiguous.
5. Assign explicitness: "explicit" if the causal/influence language is directly stated, "near_explicit" if strongly implied by the sentence structure, "inferred" if requires interpretation.
6. Confidence should reflect how clearly the relationship is expressed (0.0-1.0).
7. Do NOT hallucinate relationships not present in the text.
8. If no relationships can be extracted, return {"edges": []}.

Allowed node types: actor, technology, policy, event, process, outcome, value, belief_proposition, topic

Allowed relation types: causes, increases, reduces, enables, prevents, reinforces, undermines, increases_exposure_to, supports, opposes, associated_with, expresses, rejects, questions, reports"""


def build_extraction_prompt(window_turns: list[dict[str, Any]]) -> str:
    """Build the user prompt for a window of turns."""
    lines = ["Extract causal/influence relationships from this discussion excerpt:\n"]
    lines.append("---TRANSCRIPT EXCERPT---")
    for turn in window_turns:
        speaker = turn["speaker"]
        text = turn["text"]
        lines.append(f"[{speaker}]: {text}")
    lines.append("---END EXCERPT---\n")
    lines.append("Return a JSON object with this structure:")
    lines.append(_build_json_schema_instruction(ExtractionResult))
    lines.append("\nRespond ONLY with valid JSON. No other text.")
    return "\n".join(lines)


def create_windows(
    turns: list[dict[str, Any]],
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> list[list[dict[str, Any]]]:
    """Create sliding windows over turns grouped by table+round."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        key = (turn["table_id"], turn["round_id"])
        grouped[key].append(turn)

    windows: list[list[dict[str, Any]]] = []
    for _key, table_turns in grouped.items():
        table_turns.sort(key=lambda t: t["start_time"])
        step = max(1, window_size - overlap)
        for i in range(0, len(table_turns), step):
            window = table_turns[i : i + window_size]
            if window:
                windows.append(window)

    return windows


def validate_evidence(edge: LLMEdge, window_text: str) -> bool:
    """Check that evidence_text is a verbatim substring of the window (whitespace-normalized)."""
    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    return normalize(edge.evidence_text) in normalize(window_text)


def extract_edges_from_window(
    window_turns: list[dict[str, Any]],
) -> list[EdgeRecord]:
    """Run extraction on a single window and return validated EdgeRecords."""
    prompt = build_extraction_prompt(window_turns)
    window_text = " ".join(t["text"] for t in window_turns)

    try:
        result = generate_structured(
            prompt=prompt,
            schema=ExtractionResult,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"LLM call failed for window: {e}")
        return []

    edges: list[EdgeRecord] = []
    for llm_edge in result.edges:
        if not validate_evidence(llm_edge, window_text):
            logger.warning(
                f"Dropped edge (evidence not verbatim): {llm_edge.evidence_text[:60]}..."
            )
            continue

        first_turn = window_turns[0]
        edge = EdgeRecord(
            edge_id=str(uuid.uuid4()),
            source_node=Node(label=llm_edge.source_label, type=llm_edge.source_type),
            relation=llm_edge.relation,
            target_node=Node(label=llm_edge.target_label, type=llm_edge.target_type),
            speaker=first_turn["speaker"],
            participant_id=first_turn["participant_id"],
            discussion_id=first_turn["discussion_id"],
            room_id=first_turn["room_id"],
            table_id=first_turn["table_id"],
            round_id=first_turn["round_id"],
            turn_ids=[t["turn_id"] for t in window_turns],
            evidence_text=llm_edge.evidence_text,
            stance=llm_edge.stance,
            explicitness=llm_edge.explicitness,
            confidence=llm_edge.confidence,
            extraction_method=f"llm:{LLM_PROVIDER}/{LLM_MODEL}",
            created_at=datetime.now(timezone.utc),
        )
        edges.append(edge)

    return edges


def deduplicate_edges(edges: list[EdgeRecord]) -> list[EdgeRecord]:
    """Remove duplicate edges (same source, target, relation, evidence within same table)."""
    seen: set[tuple] = set()
    unique: list[EdgeRecord] = []
    for edge in edges:
        key = (
            edge.source_node.label.lower(),
            edge.relation,
            edge.target_node.label.lower(),
            edge.table_id,
            edge.evidence_text.lower().strip(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def run_extraction(
    turns: list[dict[str, Any]],
    table_ids: list[str] | None = None,
    output_dir: Path | None = None,
    max_workers: int | None = None,
) -> list[EdgeRecord]:
    """Run the full extraction pipeline on turns, optionally filtering by table_ids.

    Uses concurrent threads to parallelize LLM calls (I/O-bound).
    max_workers defaults to min(32, os.cpu_count() * 4) for optimal API throughput.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if table_ids:
        turns = [t for t in turns if t["table_id"] in table_ids]

    logger.info(f"Extracting from {len(turns)} turns")
    windows = create_windows(turns)
    logger.info(f"Created {len(windows)} windows")

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 4)
    logger.info(f"Using {max_workers} parallel workers")

    all_edges: list[EdgeRecord] = []
    completed = 0
    total = len(windows)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_edges_from_window, w): i for i, w in enumerate(windows)}

        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                logger.info(f"Completed {completed}/{total} windows")
            try:
                edges = future.result()
                all_edges.extend(edges)
            except Exception as e:
                logger.error(f"Window {futures[future]} failed: {e}")

    all_edges = deduplicate_edges(all_edges)
    logger.info(f"Extracted {len(all_edges)} unique edges")

    out_path = output_dir / "edges.json"
    with open(out_path, "w") as f:
        json.dump(
            [e.model_dump(mode="json") for e in all_edges],
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Written to {out_path}")

    return all_edges
