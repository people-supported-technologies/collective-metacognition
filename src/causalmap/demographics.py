"""Load demographics CSVs and join on participant_id.

Demographics are late-bound: used only at aggregation/filter time,
never during extraction.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import DEMOGRAPHICS_DIR

AFFILIATION_KEYS = (
    "U.s. political affiliation",
    "political_affiliation",
    "Political affiliation",
)
SEX_KEYS = ("Sex", "sex")
AGE_KEYS = ("Age", "age")


def _first_value(attrs: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        val = attrs.get(key, "").strip() if attrs.get(key) else ""
        if val:
            return val
    return None


def normalize_demographics(raw: dict[str, Any]) -> dict[str, str]:
    """Map raw CSV columns to stable viewer field names."""
    normalized: dict[str, str] = {}
    affiliation = _first_value(raw, AFFILIATION_KEYS)
    if affiliation:
        normalized["political_affiliation"] = affiliation
    sex = _first_value(raw, SEX_KEYS)
    if sex:
        normalized["sex"] = sex
    age = _first_value(raw, AGE_KEYS)
    if age:
        normalized["age"] = age
    ethnicity = raw.get("Ethnicity simplified", "").strip()
    if ethnicity:
        normalized["ethnicity"] = ethnicity
    return normalized


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


def load_normalized_demographics(
    demographics_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return participant_id -> normalized demographic fields for the viewer."""
    raw = load_demographics(demographics_dir)
    return {pid: normalize_demographics(attrs) for pid, attrs in raw.items()}
