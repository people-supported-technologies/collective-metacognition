"""Load the transcription.xlsx discussion sheet into a list of raw segment dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

from .config import TRANSCRIPT_PATH

DISCUSSION_COLUMNS = [
    "iteration_cycle",
    "participant_id",
    "table_id",
    "room_id",
    "speaker",
    "joined_at",
    "start_time",
    "start_timestamp",
    "end_time",
    "end_timestamp",
    "transcript",
]


def load_discussion_segments(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read the 'discussion' sheet and return one dict per segment row."""
    path = path or TRANSCRIPT_PATH
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb["discussion"]

    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() for h in next(rows)]

    col_idx = {col: header.index(col) for col in DISCUSSION_COLUMNS if col in header}

    segments: list[dict[str, Any]] = []
    for row in rows:
        text = row[col_idx["transcript"]]
        if text is None or str(text).strip() == "":
            continue
        seg: dict[str, Any] = {}
        for col, idx in col_idx.items():
            val = row[idx]
            seg[col] = str(val).strip() if val is not None else None
        segments.append(seg)

    wb.close()
    return segments
