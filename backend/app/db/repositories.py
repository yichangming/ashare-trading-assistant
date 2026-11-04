from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from app.db.models import (
    AccountRow,
    AuditLogRow,
    BacktestRunRow,
    CashFlowRow,
    CorporateActionRow,
    EventLinkRow,
    EventRevisionRow,
    EventRow,
    ExecutionRow,
    JobRunRow,
    NotificationRow,
    OrderRow,
    PlanRow,
    PortfolioSnapshotRow,
    PositionLotRow,
    PositionRow,
    SettingRow,
)
from app.domain.enums import (
    AccountMode,
    ExecutionSide,
    GlobalRiskMode,
    OrderStatus,
    PlanStatus,
    Strategy,
    SubMode,
    Template,
    ValuationStatus,
)
from app.domain.models import (
    AccountState,
    PendingExposure,
    PositionState,
    TargetState,
    TradePlan,
)


def new_id() -> str:
    return uuid4().hex


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


class AccountRepository:
    @staticmethod
    def create(
        session: Session,
        *,
        account_id: str = "default",
        name: str = "default",
        mode: AccountMode = AccountMode.PAPER,
        equity: float = 0.0,
        cash: float = 0.0,
        peak_equity: float | None = None,
        risk_config_configured: bool = False,
        max_drawdown: float = 0.15,
    ) -> AccountRow:
        row = AccountRow(
            id=account_id,
            name=name,
            mode=mode.value,
            equity=Decimal(str(equity)),
            cash=Decimal(str(cash)),
            peak_equity=Decimal(str(peak_equity if peak_equity is not None else equity)),
            initial_equity=Decimal(str(equity)),
            max_drawdown=max_drawdown,
            risk_config_configured=risk_config_configured,
            net_deposits=Decimal("0"),
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def get(session: Session, account_id: str = "default") -> AccountRow | None:
        return session.get(AccountRow, account_id)

    @staticmethod
    def ensure(
        session: Session,
        account_id: str = "default",
        **kwargs: Any,
    ) -> AccountRow:
        row = AccountRepository.get(session, account_id)
        if row is None:
            row = AccountRepository.create(session, account_id=account_id, **kwargs)
        return row

    @staticmethod
    def save_state(
        session: Session, state: AccountState, account_id: str = "default"
    ) -> AccountRow:
        row = AccountRepository.ensure(session, account_id)
        row.equity = state.equity
        row.cash = state.cash
        row.peak_equity = state.peak_equity
        row.mode = state.mode.value
        row.risk_config_configured = state.risk_config_configured
        row.trading_halted = state.trading_halted
        row.drawdown_half_latched = state.drawdown_half_latched
        row.max_drawdown = state.max_drawdown
        row.net_deposits = state.net_deposits
        row.valuation_incomplete = state.valuation_incomplete
        row.valuation_note = state.valuation_note
        row.last_synced_at = state.last_synced_at or datetime.now(UTC)
        row.last_valuation_at = state.last_valuation_at or row.last_valuation_at
        row.last_reconciled_at = (
            state.last_reconciled_at or row.last_reconciled_at
        )
        row.stale = state.stale
        session.flush()
        return row

    @staticmethod
    def to_state(
        session: Session,
        account_id: str = "default",
        current_prices: dict[str, Decimal] | None = None,
    ) -> AccountState:
        row = AccountRepository.ensure(session, account_id)
        positions = PositionRepository.list_positions(session, account_id)
        position_states: list[PositionState] = []
        prices = current_prices or {}
        for position in positions:
            price = prices.get(position.ts_code, position.current_price)
            lots = PositionRepository.list_lots(session, position.id)
            initial_shares = position.initial_shares or (
                sum(lot.shares for lot in lots if lot.add_index == 0)
                if lots
                else position.shares
            )
            add_on_count = max(
                position.add_on_count,
                max((lot.add_index for lot in lots), default=0),
            )
            strategy_value = position.primary_strategy or next(
                (lot.strategy for lot in lots if lot.strategy), None
            )
            sub_mode_value = position.sub_mode or next(
                (lot.sub_mode for lot in lots if lot.sub_mode), None
            )
            targets = [
                TargetState.model_validate(item)
                for item in (position.target_state_json or [])
            ]
            position_states.append(
                PositionState(
                    ts_code=position.ts_code,
                    sector=position.sector,
                    shares=position.shares,
                    available_shares=position.available_shares,
                    cost_price=position.cost_price,
                    current_stop=position.current_stop,
                    current_price=price,
                    template=Template.B
                    if strategy_value == Strategy.B.value
                    else Template.A,
                    strategy=Strategy(strategy_value)
                    if strategy_value
                    else None,
                    sub_mode=SubMode(sub_mode_value)
                    if sub_mode_value
                    else None,
                    primary_strategy=Strategy(strategy_value)
                    if strategy_value
                    else None,
                    strategy_version=position.strategy_version,
                    initial_r=position.initial_r,
                    initial_shares=initial_shares,
                    add_on_count=add_on_count,
                    needs_reconciliation=bool(
                        position.needs_reconciliation
                        or position.primary_strategy is None
                        or position.current_stop <= 0
                    ),
                    valuation_status=ValuationStatus(
                        position.valuation_status or "unknown"
                    ),
                    valuation_at=position.valuation_at,
                    data_time=position.valuation_at,
                    targets=targets,
                )
            )
        pending = PlanRepository.pending_exposures(session, account_id)
        return AccountState(
            equity=row.equity,
            cash=row.cash,
            peak_equity=row.peak_equity,
            positions=position_states,
            pending_plans=pending,
            trading_halted=row.trading_halted,
            drawdown_half_latched=row.drawdown_half_latched,
            mode=AccountMode(row.mode),
            risk_config_configured=row.risk_config_configured,
            max_drawdown=row.max_drawdown,
            net_deposits=row.net_deposits,
            valuation_incomplete=row.valuation_incomplete,
            valuation_note=row.valuation_note,
            last_synced_at=row.last_synced_at,
            last_valuation_at=row.last_valuation_at,
            last_reconciled_at=row.last_reconciled_at,
            stale=row.stale,
        )


class PositionRepository:
    @staticmethod
    def get(session: Session, account_id: str, ts_code: str) -> PositionRow | None:
        return session.scalar(
            select(PositionRow).where(
                PositionRow.account_id == account_id,
                PositionRow.ts_code == ts_code,
            )
        )

    @staticmethod
    def list_positions(session: Session, account_id: str = "default") -> list[PositionRow]:
        return list(
            session.scalars(
                select(PositionRow)
                .where(PositionRow.account_id == account_id)
                .order_by(PositionRow.ts_code)
            )
        )

    @staticmethod
    def list_lots(session: Session, position_id: str) -> list[PositionLotRow]:
        return list(
            session.scalars(
                select(PositionLotRow)
                .where(PositionLotRow.position_id == position_id)
                .order_by(PositionLotRow.opened_at, PositionLotRow.id)
            )
        )

    @staticmethod
    def upsert(
        session: Session,
        *,
        account_id: str,
        ts_code: str,
        sector: str = "未知",
        shares: int = 0,
        available_shares: int = 0,
        cost_price: Decimal | float = 0,
        current_stop: Decimal | float = 0,
        current_price: Decimal | float = 0,
        primary_strategy: str | None = None,
        sub_mode: str | None = None,
        strategy_version: str | None = None,
        plan_id: str | None = None,
        opened_at: datetime | None = None,
        status: str = "open",
        highest_price: Decimal | float | None = None,
        last_new_high_on: date | None = None,
        initial_r: Decimal | float | None = None,
        initial_shares: int | None = None,
        add_on_count: int | None = None,
        target_state_json: list[dict[str, Any]] | None = None,
        needs_reconciliation: bool = False,
        valuation_status: str = "unknown",
        valuation_at: datetime | None = None,
    ) -> PositionRow:
        row = PositionRepository.get(session, account_id, ts_code)
        if row is None:
            row = PositionRow(
                id=new_id(),
                account_id=account_id,
                ts_code=ts_code,
                sector=sector,
                shares=shares,
                available_shares=available_shares,
                cost_price=Decimal(str(cost_price)),
                current_stop=Decimal(str(current_stop)),
                current_price=Decimal(str(current_price)),
                primary_strategy=primary_strategy,
                sub_mode=sub_mode,
                strategy_version=strategy_version,
                plan_id=plan_id,
                opened_at=opened_at or datetime.now(UTC),
                status=status,
                highest_price=(
                    Decimal(str(highest_price))
                    if highest_price is not None
                    else Decimal(str(cost_price))
                ),
                last_new_high_on=last_new_high_on,
                initial_r=(
                    Decimal(str(initial_r)) if initial_r is not None else None
                ),
                initial_shares=(
                    int(initial_shares) if initial_shares is not None else shares
                ),
                add_on_count=add_on_count or 0,
                target_state_json=target_state_json or [],
                needs_reconciliation=needs_reconciliation,
                valuation_status=valuation_status,
                valuation_at=valuation_at,
            )
            session.add(row)
        else:
            row.sector = sector or row.sector
            row.shares = shares
            row.available_shares = available_shares
            row.cost_price = Decimal(str(cost_price))
            row.current_stop = Decimal(str(current_stop))
            row.current_price = Decimal(str(current_price))
            if primary_strategy:
                row.primary_strategy = primary_strategy
            if sub_mode:
                row.sub_mode = sub_mode
            if strategy_version:
                row.strategy_version = strategy_version
            row.plan_id = plan_id or row.plan_id
            row.status = status
            if highest_price is not None:
                row.highest_price = Decimal(str(highest_price))
            if last_new_high_on is not None:
                row.last_new_high_on = last_new_high_on
            if initial_r is not None:
                row.initial_r = Decimal(str(initial_r))
            if initial_shares is not None:
                row.initial_shares = int(initial_shares)
            if add_on_count is not None:
                row.add_on_count = int(add_on_count)
            if target_state_json is not None:
                row.target_state_json = target_state_json
            row.needs_reconciliation = bool(
                row.needs_reconciliation or needs_reconciliation
            )
            row.valuation_status = valuation_status
            row.valuation_at = valuation_at or row.valuation_at
        session.flush()
        return row

    @staticmethod
    def delete(session: Session, account_id: str, ts_code: str) -> None:
        row = PositionRepository.get(session, account_id, ts_code)
        if row is not None:
            session.delete(row)


class PlanRepository:
    @staticmethod
    def save(
        session: Session,
        plan: TradePlan,
        *,
        account_id: str = "default",
        trade_date: date | None = None,
    ) -> PlanRow:
        payload = plan.model_dump(mode="json")
        row = session.get(PlanRow, plan.id)
        strategy = (
            plan.strategy
            or (Strategy.A if plan.candidate.template == Template.A else Strategy.B)
        )
        if row is None:
            row = PlanRow(
                id=plan.id,
                account_id=account_id,
                trade_date=trade_date or plan.candidate.trade_date,
                ts_code=plan.candidate.ts_code,
                sector=plan.candidate.sector,
                strategy=strategy.value,
                sub_mode=plan.sub_mode.value if plan.sub_mode else None,
                status=plan.status.value,
                expires_on=plan.expires_on,
                expires_at=plan.expires_at,
                is_add_on=plan.position_plan.is_add_on,
                confirmation_price=plan.confirmation_price,
                max_acceptable_price=plan.max_acceptable_price,
                stop_price=plan.stop_plan.initial_stop,
                shares=plan.position_plan.shares,
                risk_amount=plan.position_plan.risk_amount,
                strategy_version=plan.strategy_version,
                data_version=plan.data_version,
                code_version=plan.code_version,
                plan_json=payload,
            )
            session.add(row)
        else:
            row.status = plan.status.value
            row.expires_on = plan.expires_on
            row.expires_at = plan.expires_at
            row.is_add_on = plan.position_plan.is_add_on
            row.stop_price = plan.stop_plan.initial_stop
            row.shares = plan.position_plan.shares
            row.risk_amount = plan.position_plan.risk_amount
            row.plan_json = payload
            row.strategy_version = plan.strategy_version
            row.data_version = plan.data_version
            row.code_version = plan.code_version
        session.flush()
        return row

    @staticmethod
    def get(session: Session, plan_id: str) -> PlanRow | None:
        return session.get(PlanRow, plan_id)

    @staticmethod
    def to_domain(row: PlanRow) -> TradePlan:
        return TradePlan.model_validate(row.plan_json)

    @staticmethod
    def list_for_date(
        session: Session, trade_date: date, status: PlanStatus | None = None
    ) -> list[PlanRow]:
        query = select(PlanRow).where(PlanRow.trade_date == trade_date)
        if status is not None:
            query = query.where(PlanRow.status == status.value)
        return list(session.scalars(query.order_by(PlanRow.created_at)))

    @staticmethod
    def list_open(session: Session, account_id: str = "default") -> list[PlanRow]:
        return list(
            session.scalars(
                select(PlanRow)
                .where(
                    PlanRow.account_id == account_id,
                    PlanRow.status.in_(
                        [
                            PlanStatus.PENDING_EXECUTION.value,
                            PlanStatus.PENDING_ENTRY.value,
                            PlanStatus.PARTIAL_FILL.value,
                            PlanStatus.OPEN.value,
                        ]
                    ),
                )
                .order_by(PlanRow.trade_date)
            )
        )

    @staticmethod
    def expire_overdue(session: Session, as_of: date) -> int:
        """Expire plans whose validity ended before ``as_of``."""
        active = (
            PlanStatus.OBSERVING.value,
            PlanStatus.PENDING_TRIGGER.value,
            PlanStatus.CONFIRMED.value,
            PlanStatus.PENDING_EXECUTION.value,
            PlanStatus.PENDING_ENTRY.value,
            PlanStatus.PARTIAL_FILL.value,
        )
        rows = list(
            session.scalars(
                select(PlanRow).where(
                    PlanRow.status.in_(active),
                    PlanRow.expires_on < as_of,
                )
            )
        )
        for row in rows:
            row.status = PlanStatus.EXPIRED.value
            payload = dict(row.plan_json or {})
            payload["cancel_reason"] = "expired"
            row.plan_json = payload
        if rows:
            session.flush()
        return len(rows)

    @staticmethod
    def expire_intraday(
        session: Session, now: datetime, account_id: str = "default"
    ) -> int:
        """Expire B2-style plans whose exact validity moment has passed."""
        active = (
            PlanStatus.PENDING_EXECUTION.value,
            PlanStatus.PENDING_ENTRY.value,
            PlanStatus.PARTIAL_FILL.value,
        )
        candidates = list(
            session.scalars(
                select(PlanRow).where(
                    PlanRow.account_id == account_id,
                    PlanRow.status.in_(active),
                    PlanRow.expires_at.is_not(None),
                )
            )
        )
        rows = []
        for row in candidates:
            expires_at = row.expires_at
            if expires_at is None:
                continue
            if expires_at.tzinfo is None and now.tzinfo is not None:
                expires_at = expires_at.replace(tzinfo=now.tzinfo)
            if expires_at <= now:
                rows.append(row)
        for row in rows:
            row.status = PlanStatus.EXPIRED.value
            payload = dict(row.plan_json or {})
            payload["cancel_reason"] = "expired"
            row.plan_json = payload
        if rows:
            session.flush()
        return len(rows)

    @staticmethod
    def active_order_remaining(
        session: Session, account_id: str, plan_id: str
    ) -> int:
        """Remaining BUY quantity still working at the broker for a plan."""
        orders = session.scalars(
            select(OrderRow).where(
                OrderRow.account_id == account_id,
                OrderRow.plan_id == plan_id,
                OrderRow.side == ExecutionSide.BUY.value,
                OrderRow.status.in_(
                    [
                        OrderStatus.PENDING.value,
                        OrderStatus.PARTIAL.value,
                        OrderStatus.CANCEL_REQUESTED.value,
                    ]
                ),
            )
        )
        return sum(
            max(0, int(order.shares) - int(order.filled_shares or 0))
            for order in orders
        )

    @staticmethod
    def pending_exposures(session: Session, account_id: str = "default") -> list[PendingExposure]:
        """Return exposure that still consumes cash/risk capacity.

        A local plan and a broker order are different facts.  Cancelling or
        expiring a plan does not cancel an order, so an active order keeps its
        remaining quantity reserved until a fill, confirmed cancellation or
        rejection is recorded.  Orders are authoritative when present.
        """
        exposures: list[PendingExposure] = []
        today = date.today()
        now = datetime.now(UTC)
        rows = list(
            session.scalars(
                select(PlanRow).where(PlanRow.account_id == account_id)
            )
        )
        plan_by_id = {row.id: row for row in rows}
        active_statuses = {
            OrderStatus.PENDING.value,
            OrderStatus.PARTIAL.value,
            OrderStatus.CANCEL_REQUESTED.value,
        }
        order_rows = list(
            session.scalars(
                select(OrderRow).where(
                    OrderRow.account_id == account_id,
                    OrderRow.side == ExecutionSide.BUY.value,
                    OrderRow.status.in_(list(active_statuses)),
                )
            )
        )
        remaining_by_plan: dict[str, int] = {}
        price_by_plan: dict[str, Decimal] = {}
        for order in order_rows:
            remaining = max(
                0, int(order.shares) - int(order.filled_shares or 0)
            )
            if remaining <= 0:
                continue
            remaining_by_plan[order.plan_id] = (
                remaining_by_plan.get(order.plan_id, 0) + remaining
            )
            price_by_plan[order.plan_id] = order.price
        active_plan_statuses = {
            PlanStatus.PENDING_EXECUTION.value,
            PlanStatus.PENDING_ENTRY.value,
            PlanStatus.PARTIAL_FILL.value,
        }
        for plan_id in sorted(set(plan_by_id) | set(remaining_by_plan)):
            row = plan_by_id.get(plan_id)
            if row is None:
                continue
            plan_active = row.status in active_plan_statuses
            if plan_active:
                if row.expires_on < today:
                    plan_active = False
                expires_at = row.expires_at
                if expires_at is not None:
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=UTC)
                    if expires_at <= now:
                        plan_active = False
            try:
                plan = PlanRepository.to_domain(row)
            except Exception:  # noqa: BLE001 - legacy payload is not exposure
                continue
            order_remaining = remaining_by_plan.get(plan_id, 0)
            if order_remaining > 0:
                # The order may outlive its plan.  Keep the exact remaining
                # quantity and the price that is actually reserved.
                entry_price = price_by_plan.get(
                    plan_id, plan.entry_plan.entry_price
                )
                if entry_price <= plan.stop_plan.initial_stop:
                    entry_price = plan.entry_plan.entry_price
                shares = order_remaining
            elif plan_active:
                filled = sum(
                    execution.shares
                    for execution in session.scalars(
                        select(ExecutionRow).where(
                            ExecutionRow.plan_id == plan.id,
                            ExecutionRow.side == ExecutionSide.BUY.value,
                        )
                    )
                )
                shares = max(0, plan.position_plan.shares - filled)
                entry_price = plan.entry_plan.entry_price
            else:
                continue
            if shares <= 0:
                continue
            exposures.append(
                PendingExposure(
                    ts_code=plan.candidate.ts_code,
                    sector=plan.candidate.sector,
                    entry_price=entry_price,
                    stop_price=plan.stop_plan.initial_stop,
                    shares=shares,
                    template=plan.candidate.template,
                    expires_on=plan.expires_on,
                    strategy=plan.strategy,
                    sub_mode=plan.sub_mode,
                )
            )
        return exposures

    @staticmethod
    def transition(
        session: Session,
        plan_id: str,
        status: PlanStatus,
        reason: str | None = None,
    ) -> PlanRow:
        from app.planning.state_machine import validate_transition

        row = session.get(PlanRow, plan_id)
        if row is None:
            raise KeyError(plan_id)
        validate_transition(PlanStatus(row.status), status)
        return PlanRepository.update_status(session, plan_id, status, reason)

    @staticmethod
    def update_status(
        session: Session,
        plan_id: str,
        status: PlanStatus,
        reason: str | None = None,
    ) -> PlanRow:
        row = session.get(PlanRow, plan_id)
        if row is None:
            raise KeyError(plan_id)
        row.status = status.value
        if reason:
            payload = dict(row.plan_json or {})
            payload["cancel_reason"] = reason
            row.plan_json = payload
        session.flush()
        return row


class ExecutionRepository:
    @staticmethod
    def add(
        session: Session,
        *,
        account_id: str,
        ts_code: str,
        side: ExecutionSide,
        price: Decimal | float,
        shares: int,
        executed_at: datetime,
        fees: Decimal | float = 0,
        plan_id: str | None = None,
        source: str = "manual",
        external_id: str | None = None,
        note: str | None = None,
        strategy: str | None = None,
        sub_mode: str | None = None,
        sector: str | None = None,
        strategy_version: str | None = None,
        batch_index: int | None = None,
        initial_r: Decimal | float | None = None,
        plan_deviation: str | None = None,
    ) -> tuple[ExecutionRow, bool]:
        if external_id:
            existing = session.scalar(
                select(ExecutionRow).where(
                    ExecutionRow.account_id == account_id,
                    ExecutionRow.external_id == external_id,
                )
            )
            if existing is not None:
                return existing, False
        row = ExecutionRow(
            id=new_id(),
            account_id=account_id,
            plan_id=plan_id,
            ts_code=ts_code,
            side=side.value,
            price=Decimal(str(price)),
            shares=int(shares),
            fees=Decimal(str(fees)),
            executed_at=executed_at,
            strategy=strategy,
            sub_mode=sub_mode,
            sector=sector,
            strategy_version=strategy_version,
            batch_index=batch_index,
            initial_r=(
                Decimal(str(initial_r)) if initial_r is not None else None
            ),
            plan_deviation=plan_deviation,
            source=source,
            external_id=external_id,
            note=note,
        )
        session.add(row)
        session.flush()
        return row, True

    @staticmethod
    def list_for_plan(session: Session, plan_id: str) -> list[ExecutionRow]:
        return list(
            session.scalars(
                select(ExecutionRow)
                .where(ExecutionRow.plan_id == plan_id)
                .order_by(ExecutionRow.executed_at)
            )
        )

    @staticmethod
    def list_for_account(session: Session, account_id: str = "default") -> list[ExecutionRow]:
        return list(
            session.scalars(
                select(ExecutionRow)
                .where(ExecutionRow.account_id == account_id)
                .order_by(ExecutionRow.executed_at)
            )
        )


class CashFlowRepository:
    @staticmethod
    def add(
        session: Session,
        *,
        account_id: str,
        flow_type: str,
        amount: Decimal | float,
        occurred_at: datetime,
        external_id: str | None = None,
        note: str | None = None,
    ) -> tuple[CashFlowRow, bool]:
        if external_id:
            existing = session.scalar(
                select(CashFlowRow).where(
                    CashFlowRow.account_id == account_id,
                    CashFlowRow.external_id == external_id,
                )
            )
            if existing is not None:
                return existing, False
        row = CashFlowRow(
            id=new_id(),
            account_id=account_id,
            flow_type=flow_type,
            amount=Decimal(str(amount)),
            occurred_at=occurred_at,
            external_id=external_id,
            note=note,
        )
        session.add(row)
        session.flush()
        return row, True


class CorporateActionRepository:
    @staticmethod
    def add(session: Session, **kwargs: Any) -> CorporateActionRow:
        row = CorporateActionRow(id=new_id(), **kwargs)
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def list_for_account(session: Session, account_id: str) -> list[CorporateActionRow]:
        return list(
            session.scalars(
                select(CorporateActionRow)
                .where(CorporateActionRow.account_id == account_id)
                .order_by(CorporateActionRow.ex_date)
            )
        )


class EventRepository:
    @staticmethod
    def upsert(
        session: Session,
        *,
        event_type: str,
        title: str,
        source: str,
        dedupe_key: str,
        ts_code: str | None = None,
        summary: str | None = None,
        source_url: str | None = None,
        published_at: datetime | None = None,
        ann_date: date | None = None,
        stage: str = "unknown",
        certainty: str = "unknown",
        direction: str = "neutral",
        materiality: float = 0.0,
        verification_status: str = "pending",
        raw_payload: dict[str, Any] | None = None,
        sub_type: str | None = None,
        retracted: bool = False,
    ) -> tuple[EventRow, bool]:
        existing = session.scalar(
            select(EventRow).where(EventRow.dedupe_key == dedupe_key)
        )
        if existing is not None:
            session.add(
                EventRevisionRow(
                    id=new_id(),
                    event_id=existing.id,
                    revision=existing.revision,
                    title=existing.title,
                    summary=existing.summary,
                    direction=existing.direction,
                    materiality=existing.materiality,
                    verification_status=existing.verification_status,
                    retracted=existing.retracted,
                    raw_payload=existing.raw_payload or {},
                )
            )
            existing.revision += 1
            existing.title = title
            existing.summary = summary
            existing.source_url = source_url
            existing.stage = stage
            existing.certainty = certainty
            existing.direction = direction
            existing.materiality = materiality
            existing.verification_status = verification_status
            existing.retracted = retracted
            existing.raw_payload = raw_payload or {}
            session.flush()
            return existing, False
        row = EventRow(
            id=new_id(),
            event_type=event_type,
            sub_type=sub_type,
            ts_code=ts_code,
            title=title,
            summary=summary,
            source=source,
            source_url=source_url,
            published_at=published_at,
            ann_date=ann_date,
            stage=stage,
            certainty=certainty,
            direction=direction,
            materiality=materiality,
            verification_status=verification_status,
            retracted=retracted,
            raw_payload=raw_payload or {},
            dedupe_key=dedupe_key,
        )
        session.add(row)
        if ts_code:
            row.links.append(
                EventLinkRow(id=new_id(), ts_code=ts_code, relation="subject")
            )
        session.flush()
        return row, True

    @staticmethod
    def list_recent(
        session: Session,
        *,
        start: date | None = None,
        end: date | None = None,
        event_type: str | None = None,
        verified_only: bool = False,
        limit: int = 200,
    ) -> list[EventRow]:
        query = select(EventRow).order_by(EventRow.ann_date.desc(), EventRow.created_at.desc())
        if start:
            query = query.where(EventRow.ann_date >= start)
        if end:
            query = query.where(EventRow.ann_date <= end)
        if event_type:
            query = query.where(EventRow.event_type == event_type)
        if verified_only:
            query = query.where(EventRow.verification_status == "verified")
        return list(session.scalars(query.limit(limit)))

    @staticmethod
    def verify(session: Session, event_id: str, status: str) -> EventRow:
        row = session.get(EventRow, event_id)
        if row is None:
            raise KeyError(event_id)
        row.verification_status = status
        session.flush()
        return row


class NotificationRepository:
    @staticmethod
    def add(
        session: Session,
        *,
        level: str,
        code: str,
        title: str,
        message: str,
        dedupe_key: str,
        trade_date: date | None = None,
    ) -> tuple[NotificationRow, bool]:
        existing = session.scalar(
            select(NotificationRow).where(NotificationRow.dedupe_key == dedupe_key)
        )
        if existing is not None:
            return existing, False
        row = NotificationRow(
            id=new_id(),
            level=level,
            code=code,
            title=title,
            message=message,
            trade_date=trade_date,
            dedupe_key=dedupe_key,
        )
        session.add(row)
        session.flush()
        return row, True

    @staticmethod
    def get_by_key(session: Session, dedupe_key: str) -> NotificationRow | None:
        return session.scalar(
            select(NotificationRow).where(NotificationRow.dedupe_key == dedupe_key)
        )

    @staticmethod
    def list_recent(
        session: Session, *, unacknowledged_only: bool = False, limit: int = 100
    ) -> list[NotificationRow]:
        priority = case(
            (NotificationRow.level.in_(("critical", "risk")), 0),
            (NotificationRow.level == "warning", 1),
            (NotificationRow.level == "info", 2),
            else_=3,
        )
        query = select(NotificationRow).order_by(
            priority, NotificationRow.created_at.desc()
        )
        if unacknowledged_only:
            query = query.where(NotificationRow.acknowledged_at.is_(None))
        return list(session.scalars(query.limit(limit)))

    @staticmethod
    def purge_older_than(session: Session, before: datetime) -> int:
        rows = list(
            session.scalars(
                select(NotificationRow).where(NotificationRow.created_at < before)
            )
        )
        for row in rows:
            session.delete(row)
        if rows:
            session.flush()
        return len(rows)

    @staticmethod
    def acknowledge(session: Session, notification_id: str) -> NotificationRow:
        row = session.get(NotificationRow, notification_id)
        if row is None:
            raise KeyError(notification_id)
        row.acknowledged_at = datetime.now(UTC)
        session.flush()
        return row


class JobRepository:
    @staticmethod
    def get_by_key(session: Session, idempotency_key: str) -> JobRunRow | None:
        return session.scalar(
            select(JobRunRow).where(JobRunRow.idempotency_key == idempotency_key)
        )

    @staticmethod
    def start(
        session: Session,
        *,
        job_name: str,
        idempotency_key: str,
        trade_date: date | None = None,
        scope: str = "full",
    ) -> JobRunRow:
        existing = session.scalar(
            select(JobRunRow).where(JobRunRow.idempotency_key == idempotency_key)
        )
        if existing is not None:
            if existing.status in ("succeeded", "running"):
                return existing
            existing.status = "running"
            existing.started_at = datetime.now(UTC)
            existing.error = None
            session.flush()
            return existing
        row = JobRunRow(
            id=new_id(),
            job_name=job_name,
            trade_date=trade_date,
            scope=scope,
            status="running",
            started_at=datetime.now(UTC),
            idempotency_key=idempotency_key,
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def finish(
        session: Session,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        data_version: str | None = None,
        strategy_version: str | None = None,
    ) -> JobRunRow:
        row = session.get(JobRunRow, job_id)
        if row is None:
            raise KeyError(job_id)
        row.status = status
        row.finished_at = datetime.now(UTC)
        row.error = error
        row.data_version = data_version
        row.strategy_version = strategy_version
        if status == "failed":
            row.retry_count += 1
        session.flush()
        return row

    @staticmethod
    def recover_running(session: Session) -> int:
        """Mark jobs left ``running`` by a previous process as failed.

        Only valid when exactly one scheduler process owns the database (the
        documented single-worker deployment).  The next scheduled/manual run
        can then retry the same idempotency key.
        """
        rows = list(
            session.scalars(
                select(JobRunRow).where(JobRunRow.status == "running")
            )
        )
        for row in rows:
            row.status = "failed"
            row.error = "recovered after process restart"
            row.finished_at = datetime.now(UTC)
            row.retry_count += 1
        if rows:
            session.flush()
        return len(rows)

    @staticmethod
    def list_recent(session: Session, limit: int = 100) -> list[JobRunRow]:
        return list(
            session.scalars(
                select(JobRunRow).order_by(JobRunRow.created_at.desc()).limit(limit)
            )
        )


class AuditRepository:
    @staticmethod
    def log(
        session: Session,
        *,
        actor: str,
        action: str,
        target: str,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> AuditLogRow:
        row = AuditLogRow(
            id=new_id(),
            actor=actor,
            action=action,
            target=target,
            before_json=before,
            after_json=after,
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def list_recent(session: Session, limit: int = 100) -> list[AuditLogRow]:
        return list(
            session.scalars(
                select(AuditLogRow).order_by(AuditLogRow.created_at.desc()).limit(limit)
            )
        )


class SettingRepository:
    @staticmethod
    def get(session: Session, key: str, default: Any = None) -> Any:
        row = session.get(SettingRow, key)
        return default if row is None else row.value_json

    @staticmethod
    def set(session: Session, key: str, value: Any) -> SettingRow:
        row = session.get(SettingRow, key)
        if row is None:
            row = SettingRow(key=key, value_json=value)
            session.add(row)
        else:
            row.value_json = value
        session.flush()
        return row


class SnapshotRepository:
    @staticmethod
    def save(
        session: Session,
        *,
        account_id: str,
        trade_date: date,
        state: AccountState,
        risk_mode: GlobalRiskMode = GlobalRiskMode.NORMAL,
        detail: dict[str, Any] | None = None,
    ) -> PortfolioSnapshotRow:
        row = session.scalar(
            select(PortfolioSnapshotRow).where(
                PortfolioSnapshotRow.account_id == account_id,
                PortfolioSnapshotRow.trade_date == trade_date,
            )
        )
        if row is None:
            row = PortfolioSnapshotRow(
                id=new_id(),
                account_id=account_id,
                trade_date=trade_date,
                equity=state.equity,
                cash=state.cash,
                positions_value=state.positions_value,
                heat=state.heat,
                drawdown=state.drawdown_pct,
                risk_mode=risk_mode.value,
                detail_json=detail or {},
            )
            session.add(row)
        else:
            row.equity = state.equity
            row.cash = state.cash
            row.positions_value = state.positions_value
            row.heat = state.heat
            row.drawdown = state.drawdown_pct
            row.risk_mode = risk_mode.value
            row.detail_json = detail or {}
        session.flush()
        return row

    @staticmethod
    def list_recent(
        session: Session, account_id: str, limit: int = 250
    ) -> list[PortfolioSnapshotRow]:
        return list(
            session.scalars(
                select(PortfolioSnapshotRow)
                .where(PortfolioSnapshotRow.account_id == account_id)
                .order_by(PortfolioSnapshotRow.trade_date.desc())
                .limit(limit)
            )
        )[::-1]


class BacktestRepository:
    @staticmethod
    def save(session: Session, row: BacktestRunRow) -> BacktestRunRow:
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def get(session: Session, run_id: str) -> BacktestRunRow | None:
        return session.get(BacktestRunRow, run_id)

    @staticmethod
    def list_recent(session: Session, limit: int = 50) -> list[BacktestRunRow]:
        return list(
            session.scalars(
                select(BacktestRunRow).order_by(BacktestRunRow.created_at.desc()).limit(limit)
            )
        )
