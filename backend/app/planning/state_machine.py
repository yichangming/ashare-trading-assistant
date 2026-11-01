"""Unified trade-plan state machine."""

from __future__ import annotations

from app.domain.enums import PlanStatus

ALLOWED: dict[PlanStatus, set[PlanStatus]] = {
    PlanStatus.DRAFT: {
        PlanStatus.OBSERVING,
        PlanStatus.PENDING_TRIGGER,
        PlanStatus.CONFIRMED,
        PlanStatus.PENDING_EXECUTION,
        PlanStatus.REJECTED,
        PlanStatus.CANCELLED,
        PlanStatus.EXPIRED,
        PlanStatus.INVALIDATED,
    },
    PlanStatus.OBSERVING: {
        PlanStatus.PENDING_TRIGGER,
        PlanStatus.INVALIDATED,
        PlanStatus.EXPIRED,
        PlanStatus.CANCELLED,
    },
    PlanStatus.PENDING_TRIGGER: {
        PlanStatus.CONFIRMED,
        PlanStatus.INVALIDATED,
        PlanStatus.EXPIRED,
        PlanStatus.CANCELLED,
    },
    PlanStatus.CONFIRMED: {
        PlanStatus.PENDING_EXECUTION,
        PlanStatus.INVALIDATED,
        PlanStatus.EXPIRED,
        PlanStatus.CANCELLED,
    },
    PlanStatus.PENDING_EXECUTION: {
        PlanStatus.PARTIAL_FILL,
        PlanStatus.OPEN,
        PlanStatus.EXPIRED,
        PlanStatus.CANCELLED,
        PlanStatus.INVALIDATED,
    },
    PlanStatus.PENDING_ENTRY: {
        PlanStatus.PARTIAL_FILL,
        PlanStatus.OPEN,
        PlanStatus.EXPIRED,
        PlanStatus.CANCELLED,
        PlanStatus.INVALIDATED,
    },
    PlanStatus.PARTIAL_FILL: {
        PlanStatus.OPEN,
        PlanStatus.EXPIRED,
        PlanStatus.CANCELLED,
        PlanStatus.INVALIDATED,
    },
    PlanStatus.OPEN: {PlanStatus.CLOSED, PlanStatus.INVALIDATED},
    PlanStatus.CLOSED: set(),
    PlanStatus.CANCELLED: set(),
    PlanStatus.EXPIRED: set(),
    PlanStatus.INVALIDATED: set(),
    PlanStatus.REJECTED: set(),
}


class InvalidTransition(ValueError):
    pass


def validate_transition(current: PlanStatus, target: PlanStatus) -> None:
    if current == target:
        return
    if target not in ALLOWED.get(current, set()):
        raise InvalidTransition(
            f"invalid plan transition: {current.value} -> {target.value}"
        )
