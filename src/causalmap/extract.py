"""Extraction pipeline: windowing, prompt assembly, LLM call, validation.

Grounded two-pass design: a controlled concept vocabulary (mined first) is injected
into the prompt so the model maps every mention onto a short, shared, polarity-neutral
concept. The speaker's directional position is carried separately as `polarity`
(edges) or as a participant->concept `StanceRecord`, so speakers are NOT graph nodes.
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
from .schemas import (
    ConceptVocabEntry,
    EdgeRecord,
    ExtractionResult,
    LLMEdge,
    LLMStance,
    Node,
    StanceRecord,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at extracting causal/influence relationships and stances from conversation transcripts.

You will be given (a) a CONTROLLED CONCEPT VOCABULARY and (b) a short excerpt from a group discussion. Extract two things:

1. EDGES: causal/influence relationships BETWEEN TWO CONCEPTS (concept -> concept).
2. STANCES: a speaker's directional position toward a SINGLE concept (e.g. supports/opposes a policy or value).

CRITICAL RULES:
1. GROUND EVERY CONCEPT to the vocabulary. For source_label / target_label / concept_label, copy the EXACT canonical label from the CONTROLLED CONCEPT VOCABULARY when one fits. Only invent a new label if no vocabulary concept is close, and then keep it SHORT (2-4 word noun phrase) and POLARITY-NEUTRAL.
2. Labels must be POLARITY-NEUTRAL: use "abortion legality" NOT "abortion should be legal"; "gun control" NOT "ban guns". Opposite opinions map to the SAME concept; the disagreement is captured by `polarity`.
3. NEVER use a speaker name/pseudonym as a concept. A speaker's belief about a single concept is a STANCE, not an edge.
4. `polarity` (supports / neutral / opposes) is the speaker's position: for an edge it is whether they affirm (supports) or deny (opposes) the relationship; for a stance it is whether they favour (supports) or are against (opposes) the concept.
5. `stance` (assertion mode): "asserted" if stated as true, "rejected" if denied, "questioned" if doubted, "reported" if attributed to others, "uncertain" if ambiguous.
6. `explicitness`: "explicit" if directly stated, "near_explicit" if strongly implied, "inferred" if interpreted.
7. evidence_text MUST be a verbatim quote copied exactly from the excerpt.
8. Do NOT hallucinate. If nothing can be extracted, return {"edges": [], "stances": []}.

Allowed node/concept types: actor, technology, policy, event, process, outcome, value, belief_proposition, topic
Allowed relation types: causes, increases, reduces, enables, prevents, reinforces, undermines, increases_exposure_to, supports, opposes, associated_with, expresses, rejects, questions, reports
Allowed polarity: supports, neutral, opposes"""

# Cap how many vocab concepts we inject per prompt (most-mentioned first) to bound tokens.
MAX_VOCAB_IN_PROMPT = 400


def build_vocab_block(vocab: list[ConceptVocabEntry]) -> str:
    """Render the controlled vocabulary as a compact, grouped, prompt-ready block."""
    if not vocab:
        return "(no vocabulary provided — use short neutral concept labels)"
    ordered = sorted(vocab, key=lambda v: -v.mention_count)[:MAX_VOCAB_IN_PROMPT]
    by_type: dict[str, list[str]] = defaultdict(list)
    for entry in ordered:
        by_type[entry.type.value].append(entry.label)
    lines: list[str] = []
    for type_value in sorted(by_type):
        labels = ", ".join(sorted(by_type[type_value]))
        lines.append(f"[{type_value}] {labels}")
    return "\n".join(lines)


def build_extraction_prompt(window_turns: list[dict[str, Any]], vocab_block: str) -> str:
    """Build the user prompt for a window of turns, with the controlled vocabulary."""
    lines = ["---CONTROLLED CONCEPT VOCABULARY (ground concepts to these labels)---"]
    lines.append(vocab_block)
    lines.append("---END VOCABULARY---\n")
    lines.append("Extract edges and stances from this discussion excerpt:\n")
    lines.append("---TRANSCRIPT EXCERPT---")
    for turn in window_turns:
        lines.append(f"[{turn['speaker']}]: {turn['text']}")
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


def _normalize_for_evidence(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def validate_evidence(evidence_text: str, window_text: str) -> bool:
    """Check that evidence_text is a verbatim substring of the window (whitespace-normalized)."""
    return _normalize_for_evidence(evidence_text) in _normalize_for_evidence(window_text)


def extract_from_window(
    window_turns: list[dict[str, Any]],
    vocab_block: str,
) -> tuple[list[EdgeRecord], list[StanceRecord]]:
    """Run extraction on a single window; return validated edges and stance records."""
    prompt = build_extraction_prompt(window_turns, vocab_block)
    window_text = " ".join(t["text"] for t in window_turns)
    first_turn = window_turns[0]
    turn_ids = [t["turn_id"] for t in window_turns]
    method = f"llm:{LLM_PROVIDER}/{LLM_MODEL}"

    try:
        result = generate_structured(
            prompt=prompt,
            schema=ExtractionResult,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"LLM call failed for window: {e}")
        return [], []

    edges: list[EdgeRecord] = []
    for llm_edge in result.edges:
        if not validate_evidence(llm_edge.evidence_text, window_text):
            logger.warning(f"Dropped edge (evidence not verbatim): {llm_edge.evidence_text[:60]}...")
            continue
        edges.append(EdgeRecord(
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
            turn_ids=turn_ids,
            evidence_text=llm_edge.evidence_text,
            stance=llm_edge.stance,
            polarity=llm_edge.polarity,
            explicitness=llm_edge.explicitness,
            confidence=llm_edge.confidence,
            extraction_method=method,
            created_at=datetime.now(timezone.utc),
        ))

    stances: list[StanceRecord] = []
    for llm_stance in result.stances:
        if not validate_evidence(llm_stance.evidence_text, window_text):
            logger.warning(f"Dropped stance (evidence not verbatim): {llm_stance.evidence_text[:60]}...")
            continue
        stances.append(StanceRecord(
            stance_id=str(uuid.uuid4()),
            concept=Node(label=llm_stance.concept_label, type=llm_stance.concept_type),
            polarity=llm_stance.polarity,
            speaker=first_turn["speaker"],
            participant_id=first_turn["participant_id"],
            discussion_id=first_turn["discussion_id"],
            room_id=first_turn["room_id"],
            table_id=first_turn["table_id"],
            round_id=first_turn["round_id"],
            turn_ids=turn_ids,
            evidence_text=llm_stance.evidence_text,
            explicitness=llm_stance.explicitness,
            confidence=llm_stance.confidence,
            extraction_method=method,
            created_at=datetime.now(timezone.utc),
        ))

    return edges, stances


def deduplicate_edges(edges: list[EdgeRecord]) -> list[EdgeRecord]:
    """Remove duplicate edges (same source, relation, target, table, evidence)."""
    seen: set[tuple] = set()
    unique: list[EdgeRecord] = []
    for edge in edges:
        key = (
            edge.source_node.label.lower(),
            edge.relation.value,
            edge.target_node.label.lower(),
            edge.table_id,
            edge.evidence_text.lower().strip(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def deduplicate_stances(stances: list[StanceRecord]) -> list[StanceRecord]:
    """Remove duplicate stance records (same concept, polarity, participant, evidence)."""
    seen: set[tuple] = set()
    unique: list[StanceRecord] = []
    for s in stances:
        key = (
            s.concept.label.lower(),
            s.polarity.value,
            s.participant_id,
            s.evidence_text.lower().strip(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def run_extraction(
    turns: list[dict[str, Any]],
    vocab: list[ConceptVocabEntry] | None = None,
    table_ids: list[str] | None = None,
    output_dir: Path | None = None,
    max_workers: int | None = None,
) -> tuple[list[EdgeRecord], list[StanceRecord]]:
    """Run grounded extraction on turns, returning concept-concept edges and stances.

    Uses concurrent threads to parallelise LLM calls (I/O-bound).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if table_ids:
        turns = [t for t in turns if t["table_id"] in table_ids]

    vocab_block = build_vocab_block(vocab or [])
    logger.info(f"Extracting from {len(turns)} turns (vocab: {len(vocab or [])} concepts)")
    windows = create_windows(turns)
    logger.info(f"Created {len(windows)} windows")

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 4)
    logger.info(f"Using {max_workers} parallel workers")

    all_edges: list[EdgeRecord] = []
    all_stances: list[StanceRecord] = []
    completed = 0
    total = len(windows)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_from_window, w, vocab_block): i
            for i, w in enumerate(windows)
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                logger.info(f"Completed {completed}/{total} windows")
            try:
                edges, stances = future.result()
                all_edges.extend(edges)
                all_stances.extend(stances)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Window {futures[future]} failed: {e}")

    all_edges = deduplicate_edges(all_edges)
    all_stances = deduplicate_stances(all_stances)
    logger.info(f"Extracted {len(all_edges)} unique edges, {len(all_stances)} unique stances")

    with open(output_dir / "edges.json", "w") as f:
        json.dump([e.model_dump(mode="json") for e in all_edges], f, indent=2, default=str)
    with open(output_dir / "stances.json", "w") as f:
        json.dump([s.model_dump(mode="json") for s in all_stances], f, indent=2, default=str)
    logger.info(f"Written edges.json and stances.json to {output_dir}")

    return all_edges, all_stances
