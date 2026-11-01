from __future__ import annotations

from app.domain.enums import Strategy, SubMode
from app.domain.models import Candidate
from app.strategies.base import (
    StrategyContext,
    base_pool,
    board_height_at,
    is_limit_up_at,
    make_candidate,
    prior_high,
    top_third,
)


def _board_candidate_index(history, observe_days: int) -> int | None:
    """Most recent first/second board in the observation window."""
    last = history.height - 2
    earliest = max(0, history.height - 2 - observe_days + 1)
    for index in range(last, earliest - 1, -1):
        if not is_limit_up_at(history, index):
            continue
        # Only the most recent limit-up matters: a 3+ board sequence must not
        # be mistaken for an earlier 2-board setup inside the window.
        height = board_height_at(history, index)
        return index if 1 <= height <= 2 else None
    return None


def _sector_hot(context: StrategyContext, row: dict, cfg) -> bool:
    sector = context.sector(str(row.get("sector") or ""))
    if sector is None:
        return False
    return sector.rank <= cfg.sector_top_n and (
        sector.limit_up_count >= cfg.sector_limit_up_min
        or sector.red_rate >= cfg.sector_red_rate_min
    )


def _score(row: dict, board_height: int, confirmed: bool) -> float:
    board = (min(max(board_height, 1), 2) / 2) * 100
    volume = min(100.0, float(row.get("vol_ratio_today_20") or 1.0) / 4 * 100)
    turnover = 100.0 if 0.05 <= float(row.get("turnover_rate") or 0) <= 0.20 else 50.0
    structure = 100.0 if confirmed else 50.0
    return 0.35 * board + 0.25 * volume + 0.20 * turnover + 0.20 * structure


def generate(context: StrategyContext) -> list[Candidate]:
    cfg = context.config.strategies.b
    universe = context.config.universe
    market = context.market.indicators
    candidates: list[Candidate] = []
    for code in context.codes:
        row = context.feature_row(code)
        history = context.history(code)
        if row is None or history is None or history.height < 30:
            continue
        if not base_pool(
            context,
            row,
            strategy=Strategy.B,
            min_amount=universe.b_min_avg_amount,
            cap_min=universe.b_market_cap_min,
            cap_max=universe.b_market_cap_max,
            price_min=3.0,
            price_max=80.0,
        ):
            continue
        if market.blow_rate > cfg.blow_rate_max:
            continue
        if market.max_board_height < cfg.max_board_min:
            continue
        if not _sector_hot(context, row, cfg):
            continue
        if float(row.get("ret_20d") or 0) > cfg.max_20d_gain:
            continue
        board_index = _board_candidate_index(history, cfg.observe_days)
        if board_index is None:
            continue
        board_height = board_height_at(history, board_index)
        board_row = history.row(board_index, named=True)
        if board_row.get("high") == board_row.get("low"):
            continue  # one-word board, not tradable next day
        board_low = float(board_row["low"])
        atr = float(row.get("atr14_raw") or 0)
        close = float(row["close"])
        low = float(row["low"])
        if atr <= 0:
            continue
        structure = close >= board_low and low >= board_low - 0.5 * atr
        if not structure:
            continue
        prev_high = prior_high(history, history.height - 1)
        confirmed = (
            prev_high is not None
            and close > prev_high
            and top_third(row)
        )
        days_since = history.height - 1 - board_index
        stage = "confirmed" if confirmed else "observing"
        score = _score(row, board_height, confirmed)
        leader = context.leader_row(code) or {}
        leader_score = float(leader.get("leader_score") or 0.0)
        total = 0.7 * leader_score + 0.3 * score if leader else score
        reasons = [
            f"前 {board_height} 板，观察第 {days_since} 日",
            f"板块 {row.get('sector')} 热度满足",
            f"结构未破，板日低点 {board_low:.2f}",
        ]
        if confirmed:
            reasons.append("收盘突破前日高点，收在振幅上 1/3")
        else:
            reasons.append("等待突破确认")
        candidate = make_candidate(
            context,
            ts_code=code,
            strategy=Strategy.B,
            sub_mode=SubMode.B1,
            setup_type="limit_up_absorption_b1",
            horizon="short",
            score=total,
            reasons=reasons,
            evidence={
                "stage": stage,
                "board_height": board_height,
                "board_low": board_low,
                "days_since_board": days_since,
                "confirmed": confirmed,
                "confirmed_high": prev_high,
            },
            feature_extras={"board_low": board_low, "days_since_board": days_since},
        )
        if candidate is not None:
            candidates.append(candidate)

        # B2 is an intraday pattern: the daily pass only publishes a watchlist
        # entry.  The intraday monitor consumes 5-minute bars and confirms.
        watch = make_candidate(
            context,
            ts_code=code,
            strategy=Strategy.B,
            sub_mode=SubMode.B2,
            setup_type="limit_up_absorption_b2_watch",
            horizon="short",
            score=total,
            reasons=reasons + ["B2 盘中观察：等待 5 分钟分歧/止跌/确认"],
            evidence={
                "stage": "watchlist",
                "board_height": board_height,
                "board_low": board_low,
                "days_since_board": days_since,
                "requires_minute_data": True,
            },
            feature_extras={"board_low": board_low, "days_since_board": days_since},
        )
        if watch is not None:
            candidates.append(watch)
    return candidates
