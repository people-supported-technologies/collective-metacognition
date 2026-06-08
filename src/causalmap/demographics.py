"""Load demographics CSVs and join on participant_id.

Demographics are late-bound: used only at aggregation/filter time,
never during extraction.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import DEMOGRAPHICS_DIR


def load_demographics(demographics_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load all CSV files in the demographics directory.

    Returns a dict keyed by participant_id -> merged demographic attributes.
    Handles varying column names for participant ID across different CSV formats.
    """
    demographics_dir = demographics_dir or DEMOGRAPHICS_DIR
    if not demographics_dir.exists():
        return {}

    PID_COLUMNS = {"participant_id", "Participant id", "Participant_id"}
    participants: dict[str, dict[str, Any]] = {}

    for csv_path in demographics_dir.glob("*.csv"):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            pid_col = next((c for c in PID_COLUMNS if c in fieldnames), None)
            if pid_col is None:
                continue
            for row in reader:
                pid = row.get(pid_col, "").strip()
                if not pid:
                    continue
                if pid not in participants:
                    participants[pid] = {}
                for key, val in row.items():
                    if key != pid_col and val and val.strip():
                        participants[pid][key] = val.strip()

    return participants
