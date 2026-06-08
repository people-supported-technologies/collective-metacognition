"""Build and persist the fixed global node registry.

The node registry is the union of all entities mentioned by anyone. It provides
stable canonical node IDs that persist across filter changes and re-runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import PROCESSED_DIR
from .schemas import CanonicalNode, EdgeRecord


def _normalize_label(label: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", label.lower().strip())


def _make_node_id(label: str) -> str:
    """Deterministic node ID from normalized label."""
    norm = _normalize_label(label)
    h = hashlib.sha256(norm.encode()).hexdigest()[:12]
    return f"node_{h}"


def build_registry(edges: list[EdgeRecord]) -> dict[str, CanonicalNode]:
    """Build the canonical node registry from all extracted edges.

    Returns a dict keyed by node_id -> CanonicalNode.
    Merges aliases when multiple surface forms map to the same normalized label.
    """
    registry: dict[str, CanonicalNode] = {}

    for edge in edges:
        for node in [edge.source_node, edge.target_node]:
            node_id = _make_node_id(node.label)

            if node_id not in registry:
                registry[node_id] = CanonicalNode(
                    node_id=node_id,
                    label=_normalize_label(node.label),
                    type=node.type,
                    aliases=[node.label],
                )
            else:
                existing = registry[node_id]
                if node.label not in existing.aliases:
                    existing.aliases.append(node.label)

    return registry


def assign_node_ids(edges: list[EdgeRecord], registry: dict[str, CanonicalNode]) -> list[EdgeRecord]:
    """Populate source_node_id and target_node_id on each edge from the registry."""
    for edge in edges:
        edge.source_node_id = _make_node_id(edge.source_node.label)
        edge.target_node_id = _make_node_id(edge.target_node.label)
    return edges


def save_registry(registry: dict[str, CanonicalNode], output_dir: Path | None = None) -> Path:
    """Persist the registry as JSON."""
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "node_registry.json"
    data = {nid: node.model_dump() for nid, node in registry.items()}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def load_registry(path: Path | None = None) -> dict[str, CanonicalNode]:
    """Load registry from JSON."""
    path = path or (PROCESSED_DIR / "node_registry.json")
    with open(path) as f:
        data = json.load(f)
    return {nid: CanonicalNode.model_validate(node) for nid, node in data.items()}
