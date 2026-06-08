"""Controlled vocabulary for node types and relation types."""

from enum import Enum


class NodeType(str, Enum):
    ACTOR = "actor"
    TECHNOLOGY = "technology"
    POLICY = "policy"
    EVENT = "event"
    PROCESS = "process"
    OUTCOME = "outcome"
    VALUE = "value"
    BELIEF_PROPOSITION = "belief_proposition"
    TOPIC = "topic"


class RelationType(str, Enum):
    CAUSES = "causes"
    INCREASES = "increases"
    REDUCES = "reduces"
    ENABLES = "enables"
    PREVENTS = "prevents"
    REINFORCES = "reinforces"
    UNDERMINES = "undermines"
    INCREASES_EXPOSURE_TO = "increases_exposure_to"
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    ASSOCIATED_WITH = "associated_with"
    EXPRESSES = "expresses"
    REJECTS = "rejects"
    QUESTIONS = "questions"
    REPORTS = "reports"


class Stance(str, Enum):
    ASSERTED = "asserted"
    REJECTED = "rejected"
    QUESTIONED = "questioned"
    REPORTED = "reported"
    UNCERTAIN = "uncertain"


class Explicitness(str, Enum):
    EXPLICIT = "explicit"
    NEAR_EXPLICIT = "near_explicit"
    INFERRED = "inferred"
