"""Account, execution, position, and corporate-action services."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.full_config import FullSystemConfig
from app.db.models import (
    AccountRow,
    CorporateActionRow,
    ExecutionRow,
    PositionLotRow,
    PositionRow,
)
from app.db.repositories import (
    AccountRepository,
    AuditRepository,
    CashFlowRepository,
    CorporateActionRepository,
    ExecutionRepository,
    PlanRepository,
    PositionRepository,
)
from app.domain.enums import (
    AccountMode,
    ExecutionSide,
    OrderStatus,
    PlanStatus,
    Strategy,
    SubMode,
)
from app.domain.models import AccountState
from app.portfolio.fees import calculate_fees


class PortfolioError(RuntimeError):
    pass


class AccountService:
    def __init__(
        self,
        session: Session,
        config: FullSystemConfig,
        account_id: str = "default",
    ):
        self.session = session
        self.config = config
        self.account_id = account_id

    def account(self) -> AccountRow:
        return AccountRepository.ensure(self.session, self.account_id)

    def configure_risk(
        self,
        *,
        equity: Decimal | float,
        cash: Decimal | float | None = None,
        max_drawdown: float = 0.15,
        risk_profile: dict[str, Any] | None = None,
    ) -> AccountRow:
        row = self.account()
        row.mode = AccountMode.REAL.value
        row.equity = Decimal(str(equity))
        row.initial_equity = Decimal(str(equity))
        row.peak_equity = max(row.peak_equity, row.equity)
        row.cash = Decimal(str(cash if cash is not None else equity))
        row.max_drawdown = max_drawdown
        row.risk_config_configured = True
        row.risk_profile_json = risk_profile or {}
        AuditRepository.log(
            self.session,
            actor="user",
            action="configure_risk",
            target=self.account_id,
            after={"equity": float(row.equity), "max_drawdown": max_drawdown},
        )
        self.session.flush()
        return row

    def refresh_available(self, as_of: date | None = None) -> None:
        day = as_of or date.today()
        for position in PositionRepository.list_positions(self.session, self.account_id):
            bought_today = sum(
                execution.shares
                for execution in self.session.scalars(
                    select(ExecutionRow).where(
                        ExecutionRow.account_id == self.account_id,
                        ExecutionRow.ts_code == position.ts_code,
                        ExecutionRow.side == ExecutionSide.BUY.value,
                    )
                )
                if execution.executed_at.date() == day
            )
            position.available_shares = max(0, position.shares - bought_today)
        self.session.flush()

    def record_execution(
        self,
        *,
        ts_code: str,
        side: ExecutionSide,
        price: Decimal | float,
        shares: int,
        executed_at: datetime,
        fees: Decimal | float | None = None,
        plan_id: str | None = None,
        source: str = "manual",
        external_id: str | None = None,
        note: str | None = None,
        sector: str = "未知",
        strategy: Strategy | None = None,
        sub_mode: SubMode | None = None,
        enforce_plan: bool = True,
    ) -> ExecutionRow:
        if shares <= 0:
            raise PortfolioError("shares must be positive")
        if side == ExecutionSide.BUY and shares % 100 != 0:
            raise PortfolioError("buy shares must be a multiple of 100")
        price_value = Decimal(str(price))
        if price_value <= 0:
            raise PortfolioError("price must be positive")
        if plan_id and enforce_plan:
            plan = PlanRepository.get(self.session, plan_id)
            if plan is not None:
                try:
                    domain = PlanRepository.to_domain(plan)
                except Exception:  # noqa: BLE001 - legacy payload
                    domain = None
                if domain is not None:
                    if executed_at.date() > domain.expires_on:
                        raise PortfolioError("plan has expired")
                    upper = domain.max_acceptable_price
                    if (
                        side == ExecutionSide.BUY
                        and upper is not None
                        and price_value > upper
                    ):
                        raise PortfolioError(
                            "price exceeds plan max_acceptable_price"
                        )
        row = self.account()
        fee_value = (
            Decimal(str(fees))
            if fees is not None
            else calculate_fees(
                side=side.value,
                price=price_value,
                shares=shares,
                costs=self.config.costs,
            ).total
        )
        execution, created = ExecutionRepository.add(
            self.session,
            account_id=self.account_id,
            ts_code=ts_code,
            side=side,
            price=price_value,
            shares=shares,
            executed_at=executed_at,
            fees=fee_value,
            plan_id=plan_id,
            source=source,
            external_id=external_id,
            note=note,
        )
        if not created:
            return execution
        if side == ExecutionSide.BUY:
            self._buy(
                row,
                ts_code=ts_code,
                price=price_value,
                shares=shares,
                fees=fee_value,
                executed_at=executed_at,
                plan_id=plan_id,
                sector=sector,
                strategy=strategy,
                sub_mode=sub_mode,
            )
        else:
            self._sell(
                row,
                ts_code=ts_code,
                price=price_value,
                shares=shares,
                fees=fee_value,
                executed_at=executed_at,
                plan_id=plan_id,
            )
        self._recompute_equity(row)
        self.session.flush()
        return execution

    def _buy(
        self,
        account: AccountRow,
        *,
        ts_code: str,
        price: Decimal,
        shares: int,
        fees: Decimal,
        executed_at: datetime,
        plan_id: str | None,
        sector: str,
        strategy: Strategy | None,
        sub_mode: SubMode | None,
    ) -> None:
        debit = price * shares + fees
        if account.cash < debit:
            raise PortfolioError("insufficient cash")
        account.cash -= debit
        position = PositionRepository.get(self.session, self.account_id, ts_code)
        prior_shares = position.shares if position else 0
        prior_cost = position.cost_price if position else Decimal("0")
        new_shares = prior_shares + shares
        new_cost = (
            (prior_cost * prior_shares + price * shares) / new_shares
            if new_shares
            else price
        )
        stop = position.current_stop if position and position.current_stop > 0 else price
        if plan_id:
            plan = PlanRepository.get(self.session, plan_id)
            if plan is not None:
                try:
                    domain = PlanRepository.to_domain(plan)
                    stop = domain.stop_plan.initial_stop
                except Exception:
                    pass
                if position and position.current_stop > 0:
                    stop = max(stop, position.current_stop)
        if position is None:
            position = PositionRepository.upsert(
                self.session,
                account_id=self.account_id,
                ts_code=ts_code,
                sector=sector,
                shares=new_shares,
                available_shares=0,
                cost_price=new_cost,
                current_stop=stop,
                current_price=price,
                primary_strategy=strategy.value if strategy else None,
                plan_id=plan_id,
                opened_at=executed_at,
            )
        else:
            position.shares = new_shares
            position.cost_price = new_cost
            position.current_stop = stop
            position.current_price = price
            if strategy and not position.primary_strategy:
                position.primary_strategy = strategy.value
            position.plan_id = plan_id or position.plan_id
        lot = PositionLotRow(
            id=uuid4().hex,
            position_id=position.id,
            strategy=strategy.value if strategy else None,
            sub_mode=sub_mode.value if sub_mode else None,
            shares=shares,
            cost_price=price,
            initial_r=None,
            opened_at=executed_at,
            add_index=len(position.lots),
            plan_id=plan_id,
        )
        self.session.add(lot)
        self.refresh_available(executed_at.date())
        if plan_id:
            self._update_plan_after_buy(plan_id)
        AuditRepository.log(
            self.session,
            actor="user",
            action="buy",
            target=ts_code,
            after={"shares": shares, "price": float(price), "fees": float(fees)},
        )

    def _sell(
        self,
        account: AccountRow,
        *,
        ts_code: str,
        price: Decimal,
        shares: int,
        fees: Decimal,
        executed_at: datetime,
        plan_id: str | None,
    ) -> None:
        position = PositionRepository.get(self.session, self.account_id, ts_code)
        if position is None:
            raise PortfolioError("position not found")
        self.refresh_available(executed_at.date())
        if shares > position.available_shares:
            raise PortfolioError("T+1 unavailable shares")
        if shares > position.shares:
            raise PortfolioError("sell exceeds position")
        account.cash += price * shares - fees
        remaining = position.shares - shares
        realized = (price - position.cost_price) * shares
        position.realized_r += float(realized)
        self._reduce_lots(position, shares)
        if remaining <= 0:
            PositionRepository.delete(self.session, self.account_id, ts_code)
            if plan_id:
                PlanRepository.update_status(
                    self.session, plan_id, PlanStatus.CLOSED, "position closed"
                )
        else:
            position.shares = remaining
            position.current_price = price
            self.refresh_available(executed_at.date())
        AuditRepository.log(
            self.session,
            actor="user",
            action="sell",
            target=ts_code,
            after={"shares": shares, "price": float(price), "fees": float(fees)},
        )

    def _reduce_lots(self, position: PositionRow, shares: int) -> None:
        remaining = shares
        for lot in PositionRepository.list_lots(self.session, position.id):
            if remaining <= 0:
                break
            reduce = min(remaining, lot.shares)
            lot.shares -= reduce
            remaining -= reduce
            if lot.shares == 0:
                self.session.delete(lot)
        if remaining > 0:
            raise PortfolioError("lot accounting mismatch")

    def _update_plan_after_buy(self, plan_id: str) -> None:
        plan = PlanRepository.get(self.session, plan_id)
        if plan is None:
            return
        filled = sum(
            execution.shares
            for execution in self.session.scalars(
                select(ExecutionRow).where(
                    ExecutionRow.plan_id == plan_id,
                    ExecutionRow.side == ExecutionSide.BUY.value,
                )
            )
        )
        status = PlanStatus.OPEN if filled >= plan.shares else PlanStatus.PARTIAL_FILL
        PlanRepository.update_status(self.session, plan_id, status)

    def _recompute_equity(self, account: AccountRow) -> None:
        positions = PositionRepository.list_positions(self.session, self.account_id)
        value = sum(
            (position.current_price * position.shares for position in positions),
            Decimal("0"),
        )
        account.equity = account.cash + value
        account.peak_equity = max(account.peak_equity, account.equity)
        account.last_synced_at = datetime.now(UTC)
        account.stale = False
        self.session.flush()

    def record_cash_flow(
        self,
        *,
        flow_type: str,
        amount: Decimal | float,
        occurred_at: datetime,
        external_id: str | None = None,
        note: str | None = None,
    ) -> bool:
        account = self.account()
        _, created = CashFlowRepository.add(
            self.session,
            account_id=self.account_id,
            flow_type=flow_type,
            amount=amount,
            occurred_at=occurred_at,
            external_id=external_id,
            note=note,
        )
        if created:
            account.cash += Decimal(str(amount))
            self._recompute_equity(account)
            AuditRepository.log(
                self.session,
                actor="user",
                action="cash_flow",
                target=self.account_id,
                after={"type": flow_type, "amount": float(amount)},
            )
        return created

    def _apply_action_values(
        self,
        position: PositionRow,
        cash_per_share: Decimal | float,
        share_ratio: float,
    ) -> Decimal:
        account = self.account()
        cash = Decimal(str(cash_per_share)) * position.shares
        if cash:
            account.cash += cash
        if share_ratio:
            new_shares = int(position.shares * (1 + share_ratio) // 100 * 100)
            if new_shares > 0:
                position.cost_price = position.cost_price * position.shares / new_shares
                position.current_stop = position.current_stop / Decimal(
                    str(1 + share_ratio)
                )
                position.shares = new_shares
                position.available_shares = new_shares
        return cash

    def apply_corporate_action(
        self,
        *,
        ts_code: str,
        ex_date: date,
        action_type: str,
        cash_per_share: Decimal | float = 0,
        share_ratio: float = 0.0,
        note: str | None = None,
        apply_now: bool = True,
    ) -> CorporateActionRow:
        row = CorporateActionRepository.add(
            self.session,
            account_id=self.account_id,
            ts_code=ts_code,
            ex_date=ex_date,
            action_type=action_type,
            cash_per_share=Decimal(str(cash_per_share)),
            share_ratio=share_ratio,
            applied=apply_now,
            applied_at=datetime.now(UTC) if apply_now else None,
            note=note,
        )
        position = PositionRepository.get(self.session, self.account_id, ts_code)
        if position is None or not apply_now:
            return row
        cash = self._apply_action_values(position, cash_per_share, share_ratio)
        self._recompute_equity(self.account())
        AuditRepository.log(
            self.session,
            actor="system",
            action="corporate_action",
            target=ts_code,
            after={"type": action_type, "cash": float(cash), "ratio": share_ratio},
        )
        return row

    def apply_due_corporate_actions(self, as_of: date) -> int:
        """Apply recorded actions whose ex-date has arrived.

        Imports may contain future ex-dates; keeping them pending avoids
        adjusting positions before the exchange does.
        """
        rows = list(
            self.session.scalars(
                select(CorporateActionRow).where(
                    CorporateActionRow.account_id == self.account_id,
                    CorporateActionRow.applied.is_(False),
                    CorporateActionRow.ex_date <= as_of,
                )
            )
        )
        if not rows:
            return 0
        for row in rows:
            position = PositionRepository.get(
                self.session, self.account_id, row.ts_code
            )
            if position is not None:
                cash = self._apply_action_values(
                    position, row.cash_per_share, row.share_ratio
                )
            else:
                cash = Decimal("0")
            row.applied = True
            row.applied_at = datetime.now(UTC)
            AuditRepository.log(
                self.session,
                actor="system",
                action="corporate_action_due",
                target=row.ts_code,
                after={
                    "type": row.action_type,
                    "cash": float(cash),
                    "ratio": row.share_ratio,
                },
            )
        self._recompute_equity(self.account())
        return len(rows)

    def mark_stale(self, as_of: date) -> bool:
        account = self.account()
        if account.last_synced_at is None or account.last_synced_at.date() < as_of:
            account.stale = True
        else:
            account.stale = False
        self.session.flush()
        return account.stale

    def state(self, current_prices: dict[str, Decimal] | None = None) -> AccountState:
        return AccountRepository.to_state(
            self.session, self.account_id, current_prices=current_prices
        )


def cancel_order(session: Session, order_id: str, reason: str) -> None:
    from app.db.models import OrderRow

    order = session.get(OrderRow, order_id)
    if order is None:
        raise PortfolioError("order not found")
    if order.status == OrderStatus.FILLED.value:
        raise PortfolioError("cannot cancel a filled order")
    if order.status == OrderStatus.CANCELLED.value:
        return
    order.status = OrderStatus.CANCELLED.value
    order.cancel_reason = reason
    AuditRepository.log(
        session,
        actor="user",
        action="cancel_order",
        target=order_id,
        after={"reason": reason},
    )
    session.flush()
