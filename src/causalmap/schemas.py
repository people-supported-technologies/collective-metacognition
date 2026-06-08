"""Pydantic models for turns, nodes, and edge records."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .ontology import Explicitness, NodeType, RelationType, Stance


class Turn(BaseModel):
    turn_id: str
    discussion_id: str
    room_id: str
    table_id: str
    round_id: str
    participant_id: str
    speaker: str
    text: str
    start_time: float
    end_time: float
    raw_segment_count: int = 1


class Node(BaseModel):
    """A node as returned by the LLM (before registry canonicalization)."""
    label: str
    type: NodeType


class CanonicalNode(BaseModel):
    """A node in the fixed global node registry."""
    node_id: str
    label: str
    type: NodeType
    aliases: list[str] = Field(default_factory=list)


class EdgeRecord(BaseModel):
    """Atomic attributed edge — the fundamental unit of storage."""
    edge_id: str
    source_node: Node
    relation: RelationType
    target_node: Node
    speaker: str
    participant_id: str
    discussion_id: str
    room_id: str
    table_id: str
    round_id: str
    turn_ids: list[str] = Field(default_factory=list)
    evidence_text: str
    stance: Stance
    explicitness: Explicitness
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str = "llm"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # After registry canonicalization these get populated
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None


class LLMEdge(BaseModel):
    """Single edge as returned by the LLM (before full EdgeRecord construction)."""
    source_label: str
    source_type: NodeType
    relation: RelationType
    target_label: str
    target_type: NodeType
    evidence_text: str
    stance: Stance
    explicitness: Explicitness
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Schema for the structured LLM output for a single window."""
    edges: list[LLMEdge] = Field(default_factory=list)
