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


class Polarity(str, Enum):
    """Speaker's signed position toward a (neutral) concept or claim.

    Decoupled from the concept label itself (claim_stance-style target+sentiment
    decomposition): the node stays neutral (e.g. "abortion legality") while the
    speaker's stance is carried here (supports / opposes / neutral). This is what
    makes a single shared node comparable across groups.
    """

    SUPPORTS = "supports"   # +1: in favour of / affirms the concept or claim
    NEUTRAL = "neutral"     #  0: mentioned without a clear directional position
    OPPOSES = "opposes"     # -1: against / denies the concept or claim


POLARITY_SIGN: dict[str, int] = {
    Polarity.SUPPORTS.value: 1,
    Polarity.NEUTRAL.value: 0,
    Polarity.OPPOSES.value: -1,
}


def polarity_sign(polarity: "Polarity | str") -> int:
    """Map a Polarity to its signed integer (+1 / 0 / -1)."""
    value = polarity.value if isinstance(polarity, Polarity) else str(polarity)
    return POLARITY_SIGN.get(value, 0)


class Explicitness(str, Enum):
    EXPLICIT = "explicit"
    NEAR_EXPLICIT = "near_explicit"
    INFERRED = "inferred"
