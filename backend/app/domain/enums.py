from enum import Enum


class Template(str, Enum):
    A = "A"
    B = "B"


class TrendRegime(str, Enum):
    ATTACK = "attack"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


class SentimentStage(str, Enum):
    FREEZE = "freeze"
    START = "start"
    FERMENT = "ferment"
    CLIMAX = "climax"
    EBB = "ebb"


class PlanStatus(str, Enum):
    DRAFT = "DRAFT"
    REJECTED = "REJECTED"
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class EntryType(str, Enum):
    PULLBACK = "pullback"
    DIVERGENCE_LOW = "divergence_low"
    CONFIRMATION = "confirmation"


class RiskOutcome(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"
