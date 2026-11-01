from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from app.core.full_config import load_full_config
from app.strategies.b2_intraday import detect_b2


def _bars() -> pl.DataFrame:
    rows = []
    prices = [
        (10.00, 10.00, 10.05, 9.98),
        (10.00, 10.00, 10.05, 9.98),
        (10.00, 10.00, 10.05, 9.98),
        (10.00, 10.00, 10.05, 9.98),
        (10.00, 10.00, 10.05, 9.98),
        (10.00, 10.00, 10.05, 9.98),
        (9.85, 9.86, 9.88, 9.84),
        (9.90, 9.86, 9.92, 9.85),
        (9.93, 9.90, 9.95, 9.89),
        (10.00, 9.95, 10.02, 9.94),
    ]
    for index, (close, open_, high, low) in enumerate(prices):
        moment = datetime(2026, 10, 30, 9, 35) + timedelta(minutes=5 * index)
        volume = 1_000_000.0
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_time": moment.isoformat(sep=" "),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "vol": volume,
                "amount": volume * close,
            }
        )
    return pl.DataFrame(rows)


def test_b2_confirms_after_two_stabilization_bars():
    signal = detect_b2("000001.SZ", _bars(), load_full_config().strategies.b)
    assert signal is not None
    assert signal.frozen_low == 9.85
    assert signal.frozen_high == 9.95
    assert signal.entry_price == 10.00
    assert signal.actionable
    assert signal.vwap_approximate is False
    assert signal.valid_until.minute == signal.signal_time.minute + 10


def test_b2_does_not_use_incomplete_future_bars():
    signal = detect_b2(
        "000001.SZ",
        _bars(),
        load_full_config().strategies.b,
        completed_until=datetime(2026, 10, 30, 10, 5),
    )
    assert signal is None


def test_b2_invalidation_resets_without_signal():
    frame = _bars()
    rows = frame.to_dicts()
    rows[9]["close"] = 9.80
    rows[9]["low"] = 9.80
    rows[9]["high"] = 9.90
    rows[9]["open"] = 9.90
    rows[9]["amount"] = 9.80 * rows[9]["vol"]
    frame = pl.DataFrame(rows)
    signal = detect_b2("000001.SZ", frame, load_full_config().strategies.b)
    assert signal is None


def test_b2_vwap_approximation_flag():
    frame = _bars()
    rows = frame.to_dicts()
    # A 1% amount/volume mismatch is enough to flag approximation without
    # changing the direction of the pattern.
    rows[0]["amount"] = rows[0]["vol"] * 9.9
    config = load_full_config().model_copy(deep=True)
    config.strategies.b.vwap_ratio_tolerance = 0.005
    signal = detect_b2("000001.SZ", pl.DataFrame(rows), config.strategies.b)
    assert signal is not None
    assert signal.vwap_approximate is True


class _CalendarProvider:
    def load_trade_cal(self, start, end):
        days = [start + timedelta(days=index) for index in range((end - start).days + 1)]
        return pl.DataFrame(
            {"cal_date": days, "is_open": [1 for _ in days]}
        )


class _MinuteSource:
    def load(self, ts_code, day):
        return _bars()


def test_b2_monitor_turns_watchlist_signal_into_plan():
    from datetime import date

    from app.db.repositories import (
        AccountRepository,
        NotificationRepository,
        PlanRepository,
        SnapshotRepository,
    )
    from app.db.session import init_db, reset_engine, session_scope
    from app.domain.enums import Strategy, SubMode, Template
    from app.domain.models import Candidate
    from app.jobs.b2 import B2Monitor
    from tests.strategy_helpers import make_bars, make_context

    reset_engine()
    init_db("sqlite://")
    day = date(2026, 10, 30)
    context = make_context(make_bars([10.0] * 40))
    candidate = Candidate(
        ts_code="000001.SZ",
        trade_date=day,
        template=Template.A,
        setup_type="limit_up_absorption_b2_watch",
        horizon="short",
        sector="测试",
        leader_rank=1,
        leader_score=80.0,
        score=80.0,
        reasons=["test"],
        features={"close": 10.0, "atr14_raw": 0.5},
        sentiment_stage=context.market.sentiment_stage,
        strategy_version="test",
        data_version="test",
        strategy=Strategy.B,
        sub_mode=SubMode.B2,
        evidence={"stage": "watchlist"},
    )
    with session_scope() as session:
        AccountRepository.ensure(session, equity=1_000_000, cash=1_000_000)
        state = AccountRepository.to_state(session)
        SnapshotRepository.save(
            session,
            account_id="default",
            trade_date=day,
            state=state,
            detail={
                "candidates": [candidate.model_dump(mode="json")],
                "regime": context.market.model_dump(mode="json"),
            },
        )
        monitor = B2Monitor(
            provider=_CalendarProvider(),
            config=load_full_config(),
            minute_source=_MinuteSource(),
        )
        result = monitor.check(
            day,
            now=datetime(
                2026, 10, 30, 10, 25, tzinfo=ZoneInfo("Asia/Shanghai")
            ),
            session=session,
        )
        assert result.signals == ["000001.SZ"]
        assert result.plans
        plans = PlanRepository.list_for_date(session, day)
        assert plans and plans[0].status == "PENDING_EXECUTION"
        assert any(
            item.code == "b2_signal_ready"
            for item in NotificationRepository.list_recent(session)
        )


def test_minute_replay_marks_missing_data_and_returns_signal():
    from app.backtest.minute import MinuteReplay

    replay = MinuteReplay(load_full_config()).run(
        datetime(2026, 10, 30).date(),
        {"000001.SZ": _bars(), "000002.SZ": pl.DataFrame()},
    )
    assert replay.missing == ["000002.SZ"]
    assert replay.signals
    assert replay.signals[0].ts_code == "000001.SZ"
    assert "T+1" in " ".join(replay.notes)
