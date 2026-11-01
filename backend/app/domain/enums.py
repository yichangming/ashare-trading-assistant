from enum import StrEnum


class Template(StrEnum):
    A = "A"
    B = "B"


class Strategy(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class SubMode(StrEnum):
    A_SHALLOW = "a_shallow"
    A_DEEP = "a_deep"
    B1 = "b1"
    B2 = "b2"
    C1 = "c1"
    C2 = "c2"
    D_UPTREND = "d_uptrend"
    D_DOWNTREND = "d_downtrend"
    E_RANGE = "e_range"
    F_EVENT = "f_event"
    F_EARNINGS = "f_earnings"
    F_BUYBACK = "f_buyback"
    F_CONTRACT = "f_contract"
    F_POLICY = "f_policy"


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
    OBSERVING = "OBSERVING"
    PENDING_TRIGGER = "PENDING_TRIGGER"
    CONFIRMED = "CONFIRMED"
    PENDING_EXECUTION = "PENDING_EXECUTION"
    PARTIAL_FILL = "PARTIAL_FILL"
    PENDING_ENTRY = "PENDING_ENTRY"  # legacy alias for PENDING_EXECUTION
    REJECTED = "REJECTED"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    INVALIDATED = "INVALIDATED"


class EntryType(StrEnum):
    PULLBACK = "pullback"
    DIVERGENCE_LOW = "divergence_low"
    CONFIRMATION = "confirmation"
    BREAKOUT = "breakout"
    RECLAIM = "reclaim"


class RiskOutcome(StrEnum):
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"


class PermissionMode(StrEnum):
    NORMAL = "normal"
    REDUCED = "reduced"
    PAUSED = "paused"


class GlobalRiskMode(StrEnum):
    NORMAL = "normal"
    PAUSE_NEW = "pause_new"
    REDUCE = "reduce"
    CLEAR = "clear"


class AccountMode(StrEnum):
    REAL = "real"
    PAPER = "paper"
    DEMO = "demo"


class ExecutionSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class EventType(StrEnum):
    EARNINGS = "earnings"
    BUYBACK = "buyback"
    CONTRACT = "contract"
    POLICY = "policy"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class EventDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EventStage(StrEnum):
    PLAN = "plan"
    IMPLEMENTATION = "implementation"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class EventCertainty(StrEnum):
    INTENT = "intent"
    FORMAL = "formal"
    UNKNOWN = "unknown"


class NotificationLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    RISK = "risk"
    CRITICAL = "critical"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
