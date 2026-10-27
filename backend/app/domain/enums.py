from enum import StrEnum


class Template(StrEnum):
    A = "A"
    B = "B"


class TrendRegime(StrEnum):
    ATTACK = "attack"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


class SentimentStage(StrEnum):
    FREEZE = "freeze"
    START = "start"
    FERMENT = "ferment"
    CLIMAX = "climax"
    EBB = "ebb"


class PlanStatus(StrEnum):
    DRAFT = "DRAFT"
    REJECTED = "REJECTED"
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class EntryType(StrEnum):
    PULLBACK = "pullback"
    DIVERGENCE_LOW = "divergence_low"
    CONFIRMATION = "confirmation"


class RiskOutcome(StrEnum):
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"
