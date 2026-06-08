"""Utterance reconstruction and pseudonymization.

Merges consecutive same-speaker ASR segments within a gap threshold into
coherent utterances, pseudonymizes speaker names, and writes canonical
turns JSON per table.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import PROCESSED_DIR, UTTERANCE_GAP_THRESHOLD_S


def _speaker_pseudonym(speaker: str, salt: str = "causalmap_poc") -> str:
    """Deterministic pseudonym: Speaker_<short hash>."""
    h = hashlib.sha256(f"{salt}:{speaker}".encode()).hexdigest()[:8]
    return f"Speaker_{h}"


def reconstruct_utterances(
    segments: list[dict[str, Any]],
    gap_threshold: float = UTTERANCE_GAP_THRESHOLD_S,
) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker segments within gap_threshold into turns.

    Groups by (table_id, iteration_cycle) first, sorts by start_time,
    then merges consecutive segments from the same speaker if the gap
    between end of previous and start of next is <= gap_threshold.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        key = (seg["table_id"], seg["iteration_cycle"])
        grouped[key].append(seg)

    turns: list[dict[str, Any]] = []

    for (_table_id, _round_id), table_segs in grouped.items():
        table_segs.sort(key=lambda s: float(s["start_time"]))

        current: dict[str, Any] | None = None
        current_seg_ids: list[int] = []
        current_texts: list[str] = []

        for i, seg in enumerate(table_segs):
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            speaker = seg["speaker"]

            if current is None:
                current = seg
                current_texts = [seg["transcript"]]
                current_seg_ids = [i]
            elif (
                speaker == current["speaker"]
                and (start - float(current["_end_time"])) <= gap_threshold
            ):
                current_texts.append(seg["transcript"])
                current_seg_ids.append(i)
                current["_end_time"] = seg["end_time"]
            else:
                turns.append(_finalize_turn(current, current_texts, current_seg_ids))
                current = seg
                current_texts = [seg["transcript"]]
                current_seg_ids = [i]

            if current is not None:
                current["_end_time"] = seg["end_time"]

        if current is not None:
            turns.append(_finalize_turn(current, current_texts, current_seg_ids))

    return turns


def _finalize_turn(
    first_seg: dict[str, Any],
    texts: list[str],
    seg_indices: list[int],
) -> dict[str, Any]:
    """Build a canonical turn dict from merged segments."""
    return {
        "turn_id": str(uuid.uuid4()),
        "discussion_id": "e5e2d424-3022-4174-982b-4870b1f9029c",
        "room_id": first_seg["room_id"],
        "table_id": first_seg["table_id"],
        "round_id": first_seg["iteration_cycle"],
        "participant_id": first_seg["participant_id"],
        "speaker": first_seg["speaker"],
        "text": " ".join(t.strip() for t in texts if t.strip()),
        "start_time": float(first_seg["start_time"]),
        "end_time": float(first_seg["_end_time"]),
        "raw_segment_count": len(seg_indices),
    }


def pseudonymize_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace speaker names with deterministic pseudonyms."""
    for turn in turns:
        turn["speaker_original"] = turn["speaker"]
        turn["speaker"] = _speaker_pseudonym(turn["speaker"])
    return turns


def write_canonical_turns(
    turns: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> Path:
    """Write all turns as a single JSON file, sorted by table then time."""
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    turns_sorted = sorted(turns, key=lambda t: (t["table_id"], t["round_id"], t["start_time"]))

    out_path = output_dir / "canonical_turns.json"
    with open(out_path, "w") as f:
        json.dump(turns_sorted, f, indent=2, default=str)

    return out_path
