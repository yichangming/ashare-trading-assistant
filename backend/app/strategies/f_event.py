from __future__ import annotations

from datetime import date, datetime

from app.domain.enums import Strategy, SubMode
from app.domain.models import Candidate
from app.strategies.base import StrategyContext, make_candidate
from app.strategies.c_platform import generate as generate_c

_EVENT_SUBMODE = {
    "earnings": SubMode.F_EARNINGS,
    "buyback": SubMode.F_BUYBACK,
    "contract": SubMode.F_CONTRACT,
    "policy": SubMode.F_POLICY,
}


def _event_date(event: dict) -> date | None:
    value = (
        event.get("first_seen_at")
        or event.get("ann_date")
        or event.get("published_at")
    )
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def generate(context: StrategyContext) -> list[Candidate]:
    cfg = context.config.strategies.f
    if not context.events_available:
        return []
    c_candidates = {
        candidate.ts_code: candidate
        for candidate in generate_c(context)
        if candidate.strategy == Strategy.C
    }
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for code in context.codes:
        events = context.events_for(code)
        if not events:
            continue
        if any(
            event.get("direction") == "negative"
            and event.get("verification_status") == "verified"
            and not event.get("retracted")
            for event in events
        ):
            continue
        c_candidate = c_candidates.get(code)
        if c_candidate is None:
            continue
        if cfg.price_mode == "c1" and c_candidate.sub_mode != SubMode.C1:
            continue
        if cfg.price_mode == "c2" and c_candidate.sub_mode != SubMode.C2:
            continue
        seen_types: set[SubMode] = set()
        for event in events:
            if event.get("retracted"):
                continue
            if event.get("verification_status") != "verified":
                continue
            if cfg.require_verified and not event.get("verified", True):
                continue
            if event.get("ambiguous") and not cfg.allow_ambiguous:
                continue
            if event.get("direction") != "positive":
                continue
            if float(event.get("materiality") or 0.0) < cfg.min_materiality:
                continue
            event_day = _event_date(event)
            if event_day is None:
                continue
            if event_day > context.as_of:
                continue
            trading_age = context.trading_days_since(event_day)
            if trading_age > cfg.event_window_days:
                continue
            dedupe = (str(event.get("event_id") or event.get("dedupe_key")), code)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            sub_mode = _EVENT_SUBMODE.get(
                str(event.get("event_type") or ""), SubMode.F_EVENT
            )
            if sub_mode in seen_types:
                continue
            seen_types.add(sub_mode)
            row = context.feature_row(code)
            if row is None or c_candidate.sub_mode is None:
                continue
            c_mode = c_candidate.sub_mode.value
            stage = c_candidate.evidence.get("stage", "observing")
            reasons = [
                f"事件：{event.get('title') or event.get('event_type')}",
                f"来源 {event.get('source')}，核验通过，方向正面",
                *c_candidate.reasons,
            ]
            candidate = make_candidate(
                context,
                ts_code=code,
                strategy=Strategy.F,
                sub_mode=sub_mode,
                setup_type=f"event_{c_mode}",
                horizon="mid",
                score=min(
                    100.0,
                    c_candidate.score
                    + min(10.0, float(event.get("materiality") or 0) * 10),
                ),
                reasons=reasons,
                evidence={
                    **c_candidate.evidence,
                    "event_id": event.get("event_id") or event.get("dedupe_key"),
                    "event_type": event.get("event_type"),
                    "event_stage": event.get("stage"),
                    "event_certainty": event.get("certainty"),
                    "event_source": event.get("source"),
                    "event_ann_date": event.get("ann_date"),
                    "first_seen_at": event.get("first_seen_at"),
                    "price_mode": c_mode,
                    "confirmed": c_candidate.evidence.get("confirmed", False),
                    "stage": stage,
                },
                feature_extras=c_candidate.features,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates
