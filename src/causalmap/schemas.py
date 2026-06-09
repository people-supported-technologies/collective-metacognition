"""Pydantic models for turns, nodes, and edge records."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .ontology import Explicitness, NodeType, Polarity, RelationType, Stance


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
    """A node as returned by the LLM (before registry canonicalization).

    `label` should be a short, polarity-neutral concept (e.g. "abortion legality"),
    NOT a full sentence. `concept_id` is the vocabulary entry the LLM grounded this
    mention to (may be None if the model proposed a new concept).
    """
    label: str
    type: NodeType
    concept_id: Optional[str] = None


class CanonicalNode(BaseModel):
    """A node in the fixed global node registry (one per controlled-vocab concept)."""
    node_id: str
    label: str
    type: NodeType
    aliases: list[str] = Field(default_factory=list)


class ConceptVocabEntry(BaseModel):
    """A single entry in the controlled concept vocabulary (the shared backbone).

    Labels are deliberately neutral noun phrases so opposite positions collapse onto
    one comparable node, with the stance carried separately as Polarity.
    """
    concept_id: str
    label: str
    type: NodeType
    aliases: list[str] = Field(default_factory=list)
    mention_count: int = 0


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
    polarity: Polarity = Polarity.NEUTRAL
    explicitness: Explicitness
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str = "llm"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # After registry canonicalization these get populated
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None


class StanceRecord(BaseModel):
    """Atomic participant -> concept stance (the overlay on the causal graph).

    Represents a speaker taking a directional position toward a single concept
    (e.g. supports "gun control"), without requiring a second concept. This
    replaces the old pattern of materialising the speaker as a graph node via
    `expresses`/`rejects` edges.
    """
    stance_id: str
    concept: Node
    polarity: Polarity
    speaker: str
    participant_id: str
    discussion_id: str
    room_id: str
    table_id: str
    round_id: str
    turn_ids: list[str] = Field(default_factory=list)
    evidence_text: str
    explicitness: Explicitness
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str = "llm"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    concept_node_id: Optional[str] = None


class LLMEdge(BaseModel):
    """Single concept->concept edge as returned by the LLM."""
    source_label: str
    source_type: NodeType
    source_concept_id: Optional[str] = None
    relation: RelationType
    target_label: str
    target_type: NodeType
    target_concept_id: Optional[str] = None
    evidence_text: str
    stance: Stance
    polarity: Polarity = Polarity.NEUTRAL
    explicitness: Explicitness
    confidence: float = Field(ge=0.0, le=1.0)


class LLMStance(BaseModel):
    """A speaker's directional position toward a single concept, as returned by the LLM."""
    concept_label: str
    concept_type: NodeType
    concept_id: Optional[str] = None
    polarity: Polarity
    evidence_text: str
    explicitness: Explicitness
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Structured LLM output for a single window: concept-concept edges + stances."""
    edges: list[LLMEdge] = Field(default_factory=list)
    stances: list[LLMStance] = Field(default_factory=list)
