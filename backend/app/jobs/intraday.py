"""Intraday position protection monitor.

The monitor only raises notifications; it never places orders.  If realtime
quotes are unavailable it records a data-health notification and does not
pretend a daily/close price is live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

from app.db.repositories import (
    NotificationRepository,
    PositionRepository,
)
from app.db.session import session_scope

SHANGHAI = ZoneInfo("Asia/Shanghai")


class QuoteAdapter(Protocol):
    def get_quote(self, ts_code: str) -> tuple[float, datetime] | None: ...


@dataclass
class TushareQuoteAdapter:
    """Free Tushare realtime quote; may return None outside market hours."""

    def get_quote(self, ts_code: str) -> tuple[float, datetime] | None:
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
        moment = datetime.now(SHANGHAI)
        for key in ("TIME", "time", "trade_time"):
            value = row.get(key)
            if value:
                try:
                    moment = datetime.fromisoformat(
                        f"{date.today().isoformat()} {str(value)}"
                    ).replace(tzinfo=SHANGHAI)
                    break
                except ValueError:
                    pass
        return price, moment


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
    ):
        self.quote_adapter = quote_adapter
        self.account_id = account_id
        self.b2 = b2

    def check(
        self, as_of: date | None = None, *, now: datetime | None = None
    ) -> dict:
        day = as_of or date.today()
        current = now or datetime.now(SHANGHAI)
        if not is_trading_session(current):
            return {"checked": 0, "alerts": 0, "degraded": False}
        alerts = 0
        checked = 0
        degraded = False
        with session_scope() as session:
            positions = PositionRepository.list_positions(session, self.account_id)
            for position in positions:
                quote = self.quote_adapter.get_quote(position.ts_code)
                if quote is None:
                    degraded = True
                    NotificationRepository.add(
                        session,
                        level="warning",
                        code="intraday_data_degraded",
                        title="盘中实时数据不可用",
                        message=(
                            "未取得实时报价；系统不会用日线价格冒充实时，"
                            "请检查实时/分钟数据源。"
                        ),
                        dedupe_key=f"intraday-degraded:{day.isoformat()}",
                        trade_date=day,
                    )
                    break
                checked += 1
                price, moment = quote
                position.current_price = Decimal(str(price))
                if price <= float(position.current_stop):
                    alerts += 1
                    NotificationRepository.add(
                        session,
                        level="critical",
                        code="hard_stop_breached",
                        title=f"{position.ts_code} 触及硬止损",
                        message=(
                            f"现价 {price:.2f} ≤ 保护线 "
                            f"{position.current_stop:.2f}；请人工复核执行。"
                        ),
                        dedupe_key=(
                            f"hard-stop:{day.isoformat()}:{position.ts_code}:{price:.2f}"
                        ),
                        trade_date=day,
                    )
                elif price <= float(position.current_stop) * 1.02:
                    alerts += 1
                    NotificationRepository.add(
                        session,
                        level="warning",
                        code="near_stop",
                        title=f"{position.ts_code} 接近保护线",
                        message=(
                            f"现价 {price:.2f} 距保护线 "
                            f"{position.current_stop:.2f} 不足 2%。"
                        ),
                        dedupe_key=(
                            f"near-stop:{day.isoformat()}:{position.ts_code}"
                        ),
                        trade_date=day,
                    )
            if self.b2 is not None:
                try:
                    self.b2.check(day, now=current, session=session)
                except Exception as exc:  # noqa: BLE001 - stop monitor must run
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
        return {"checked": checked, "alerts": alerts, "degraded": degraded}
