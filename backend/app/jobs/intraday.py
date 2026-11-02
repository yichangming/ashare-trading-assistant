"""Intraday position protection monitor.

The monitor only raises notifications; it never places orders.  Quote data is
validated for date, timezone, market time and fetch time.  A stale or invalid
quote does not update the position price and never triggers a stop alert.
One failed symbol never stops the remaining positions from being checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol
from zoneinfo import ZoneInfo

from app.core.calendar import TradingCalendar
from app.core.full_config import FullSystemConfig, load_full_config
from app.data.interfaces import MarketDataProvider
from app.db.repositories import PositionRepository
from app.db.session import session_scope
from app.domain.enums import ExitSession, ValuationStatus
from app.portfolio.decisions import PositionDecisionService
from app.portfolio.market import PricePoint

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Quote:
    price: float
    quote_time: datetime | None
    fetched_at: datetime
    source: str = "realtime"
    delayed: bool = False


class QuoteAdapter(Protocol):
    def get_quote(self, ts_code: str) -> Quote | tuple[float, datetime] | None: ...


@dataclass
class TushareQuoteAdapter:
    """Free Tushare realtime quote; may return None outside market hours."""

    def get_quote(self, ts_code: str) -> Quote | None:
        fetched_at = datetime.now(SHANGHAI)
        try:
            import tushare as ts

            frame = ts.realtime_quote(ts_code=ts_code)
        except Exception:
            return None
        if frame is None or getattr(frame, "empty", True):
            return None
        row = frame.iloc[0].to_dict()
        price = None
        for key in ("PRICE", "price", "CLOSE", "close"):
            value = row.get(key)
            if value is not None:
                try:
                    price = float(value)
                    break
                except (TypeError, ValueError):
                    continue
        if price is None or price <= 0:
            return None
        quote_time = _quote_time_from_row(row)
        return Quote(
            price=price,
            quote_time=quote_time,
            fetched_at=fetched_at,
            source="tushare_realtime",
        )


def _quote_time_from_row(row: dict) -> datetime | None:
    value = None
    for key in ("TIME", "time", "trade_time"):
        if row.get(key):
            value = row.get(key)
            break
    if value is None:
        return None
    day_value = None
    for key in ("DATE", "date", "trade_date"):
        if row.get(key):
            day_value = str(row.get(key))
            break
    try:
        parsed_time = datetime.fromisoformat(str(value)).time()
    except ValueError:
        try:
            parsed_time = datetime.strptime(str(value), "%H:%M:%S").time()
        except ValueError:
            return None
    if day_value:
        try:
            day = date.fromisoformat(day_value[:10])
        except ValueError:
            try:
                day = datetime.strptime(day_value[:8], "%Y%m%d").date()
            except ValueError:
                return None
    else:
        day = datetime.now(SHANGHAI).date()
    return datetime.combine(day, parsed_time, tzinfo=SHANGHAI)


def is_trading_session(moment: datetime | None = None) -> bool:
    current = moment or datetime.now(SHANGHAI)
    if current.weekday() >= 5:
        return False
    clock = current.time()
    return (
        time(9, 30) <= clock <= time(11, 30)
        or time(13, 0) <= clock <= time(15, 0)
    )


class IntradayMonitor:
    def __init__(
        self,
        quote_adapter: QuoteAdapter,
        account_id: str = "default",
        b2=None,
        *,
        minute_source=None,
        calendar_provider: MarketDataProvider | None = None,
        provider: MarketDataProvider | None = None,
        config: FullSystemConfig | None = None,
    ):
        self.quote_adapter = quote_adapter
        self.account_id = account_id
        self.b2 = b2
        self.minute_source = minute_source
        self.calendar_provider = calendar_provider
        self.provider = provider
        self.config = config or load_full_config()

    def _normalise(self, raw, now: datetime) -> Quote | None:
        if raw is None:
            return None
        if isinstance(raw, Quote):
            return raw
        if isinstance(raw, tuple) and len(raw) >= 2:
            price, moment = raw[0], raw[1]
            try:
                return Quote(
                    price=float(price),
                    quote_time=moment,
                    fetched_at=now,
                    source="adapter",
                )
            except (TypeError, ValueError):
                return None
        return None

    def _trading_day(self, day: date) -> bool:
        if self.calendar_provider is None:
            return day.weekday() < 5
        try:
            frame = self.calendar_provider.load_trade_cal(day, day)
        except Exception:  # noqa: BLE001 - monitor must keep running
            return day.weekday() < 5
        if frame.is_empty():
            return day.weekday() < 5
        calendar = TradingCalendar.from_trade_cal(frame)
        return calendar.is_trading_day(day)

    def _minute_quote(
        self, ts_code: str, day: date, now: datetime
    ) -> Quote | None:
        if not self.config.intraday.minute_fallback or self.minute_source is None:
            return None
        try:
            bars = self.minute_source.load(ts_code, day)
        except Exception:  # noqa: BLE001 - data is reported, not fabricated
            return None
        if bars is None or bars.is_empty() or "trade_time" not in bars.columns:
            return None
        rows = bars.sort("trade_time").to_dicts()
        selected = None
        for row in rows:
            moment = row.get("trade_time")
            if isinstance(moment, str):
                try:
                    moment = datetime.fromisoformat(moment)
                except ValueError:
                    continue
            if not isinstance(moment, datetime):
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=SHANGHAI)
            # Tushare labels a completed bar by its end time.
            if moment <= now:
                selected = (row, moment)
        if selected is None:
            return None
        row, moment = selected
        try:
            price = float(row.get("close") or 0)
        except (TypeError, ValueError):
            return None
        if price <= 0:
            return None
        return Quote(
            price=price,
            quote_time=moment,
            fetched_at=now,
            source="minute_delayed",
            delayed=True,
        )

    def _validate(
        self, quote: Quote | None, day: date, now: datetime
    ) -> tuple[ValuationStatus, tuple[str, ...], bool]:
        if quote is None:
            return ValuationStatus.MISSING, ("未取得实时报价",), False
        if quote.quote_time is None:
            return (
                ValuationStatus.UNKNOWN,
                ("报价缺少可靠行情时间",),
                False,
            )
        moment = quote.quote_time
        if moment.tzinfo is None:
            return (
                ValuationStatus.UNKNOWN,
                ("报价时间缺少时区，无法确认新鲜度",),
                False,
            )
        if quote.fetched_at.tzinfo is None:
            return (
                ValuationStatus.UNKNOWN,
                ("报价获取时间缺少时区",),
                False,
            )
        moment = moment.astimezone(SHANGHAI)
        fetched_at = quote.fetched_at.astimezone(SHANGHAI)
        if (
            fetched_at - now
        ).total_seconds() > self.config.intraday.max_future_skew_seconds:
            return (
                ValuationStatus.UNKNOWN,
                ("报价获取时间在未来",),
                False,
            )
        if (
            moment - fetched_at
        ).total_seconds() > self.config.intraday.max_future_skew_seconds:
            return (
                ValuationStatus.UNKNOWN,
                ("报价获取时间早于行情时间",),
                False,
            )
        future_skew = (
            now - moment
        ).total_seconds()
        if future_skew < -self.config.intraday.max_future_skew_seconds:
            return (
                ValuationStatus.UNKNOWN,
                (f"报价时间在未来：{moment.isoformat()}",),
                False,
            )
        if moment.date() != day:
            status = (
                ValuationStatus.STALE
                if moment.date() < day
                else ValuationStatus.UNKNOWN
            )
            return (
                status,
                (
                    f"报价日期 {moment.date().isoformat()} 与监控日期 "
                    f"{day.isoformat()} 不一致",
                ),
                False,
            )
        age = (now - moment).total_seconds()
        limit = (
            self.config.intraday.max_delayed_seconds
            if quote.delayed
            else self.config.intraday.max_quote_age_seconds
        )
        if age > limit:
            return (
                ValuationStatus.STALE,
                (f"报价已过期 {age:.0f}s（阈值 {limit}s）",),
                True,
            )
        return ValuationStatus.OK, (), True

    def check(
        self, as_of: date | None = None, *, now: datetime | None = None
    ) -> dict:
        day = as_of or datetime.now(SHANGHAI).date()
        current = now or datetime.now(SHANGHAI)
        if not self._trading_day(day) or not is_trading_session(current):
            return {
                "checked": 0,
                "alerts": 0,
                "degraded": False,
                "skipped": True,
                "positions": [],
            }
        points: dict[str, PricePoint] = {}
        health: list[dict] = []
        with session_scope() as session:
            positions = PositionRepository.list_positions(session, self.account_id)
            for position in positions:
                quote = self._normalise(
                    self.quote_adapter.get_quote(position.ts_code), current
                )
                if quote is None:
                    quote = self._minute_quote(position.ts_code, day, current)
                status, issues, usable = self._validate(quote, day, current)
                price = float(position.current_price or 0) if quote is None else quote.price
                points[position.ts_code] = PricePoint(
                    ts_code=position.ts_code,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    price=price,
                    data_time=quote.quote_time if quote else None,
                    status=status,
                    issues=issues,
                    source=quote.source if quote else "none",
                    delayed=bool(quote and quote.delayed),
                )
                health.append(
                    {
                        "ts_code": position.ts_code,
                        "status": status.value,
                        "issues": list(issues),
                        "source": quote.source if quote else "none",
                        "data_time": quote.quote_time.isoformat()
                        if quote and quote.quote_time
                        else None,
                        "fetched_at": quote.fetched_at.isoformat()
                        if quote
                        else None,
                        "usable": usable,
                    }
                )
            decisions = PositionDecisionService(
                session,
                self.config,
                account_id=self.account_id,
                provider=self.provider,
            ).evaluate_all(
                day,
                points=points,
                session_type=ExitSession.INTRADAY,
                now=current,
                persist=True,
                notify=True,
            )
            if self.b2 is not None:
                try:
                    self.b2.check(day, now=current, session=session)
                except Exception as exc:  # noqa: BLE001 - stop monitor must run
                    from app.db.repositories import NotificationRepository

                    NotificationRepository.add(
                        session,
                        level="warning",
                        code="b2_monitor_failed",
                        title="B2 盘中监控失败",
                        message=str(exc),
                        dedupe_key=f"b2-monitor-failed:{day.isoformat()}",
                        trade_date=day,
                    )
            session.flush()
        checked = sum(1 for item in health if item["usable"])
        alerts = sum(
            1
            for decision in decisions
            if decision.action.value in ("exit", "reduce", "data_anomaly")
        )
        degraded = any(not item["usable"] for item in health)
        return {
            "checked": checked,
            "alerts": alerts,
            "degraded": degraded,
            "skipped": False,
            "positions": health,
        }
