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
    OrderRow,
    PlanRow,
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
    ValuationStatus,
)
from app.domain.models import AccountState
from app.portfolio.fees import calculate_fees
from app.portfolio.market import PricePoint


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
        row.last_reconciled_at = datetime.now(UTC)
        row.stale = False
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
        enforce_plan: bool | None = None,
        strict_plan: bool = False,
    ) -> ExecutionRow:
        """Record a real fill.

        Plan linkage is attribution, not permission to rewrite history.  A
        known plan supplies strategy/sub-mode/sector/version and the initial
        stop; price or expiry deviations are persisted for audit.  Pass
        ``strict_plan=True`` when a caller wants the old fail-closed behavior.
        """
        if shares <= 0:
            raise PortfolioError("shares must be positive")
        if side == ExecutionSide.BUY and shares % 100 != 0:
            raise PortfolioError("buy shares must be a multiple of 100")
        price_value = Decimal(str(price))
        if price_value <= 0:
            raise PortfolioError("price must be positive")
        if external_id:
            existing = self.session.scalar(
                select(ExecutionRow).where(
                    ExecutionRow.account_id == self.account_id,
                    ExecutionRow.external_id == external_id,
                )
            )
            if existing is not None:
                return existing

        if enforce_plan is not None:
            strict_plan = strict_plan or bool(enforce_plan)

        plan_row = PlanRepository.get(self.session, plan_id) if plan_id else None
        domain = None
        if plan_row is not None:
            if plan_row.account_id != self.account_id:
                raise PortfolioError("plan belongs to a different account")
            if plan_row.ts_code != ts_code:
                raise PortfolioError("plan does not match execution security")
            try:
                domain = PlanRepository.to_domain(plan_row)
            except Exception:  # noqa: BLE001 - legacy payload
                domain = None
            if domain is not None:
                strategy = strategy or domain.strategy or (
                    Strategy(plan_row.strategy) if plan_row.strategy else None
                )
                sub_mode = sub_mode or domain.sub_mode or (
                    SubMode(plan_row.sub_mode) if plan_row.sub_mode else None
                )
                sector = plan_row.sector or sector
            else:
                strategy = strategy or (
                    Strategy(plan_row.strategy) if plan_row.strategy else None
                )
                sub_mode = sub_mode or (
                    SubMode(plan_row.sub_mode) if plan_row.sub_mode else None
                )
                sector = plan_row.sector or sector

        deviations: list[str] = []
        if domain is not None:
            if executed_at.date() > domain.expires_on:
                deviations.append(
                    f"成交晚于计划有效期 {domain.expires_on.isoformat()}"
                )
            upper = domain.max_acceptable_price
            if (
                side == ExecutionSide.BUY
                and upper is not None
                and price_value > upper
            ):
                deviations.append(
                    f"成交价 {price_value} 高于最高接受价 {upper}"
                )
            assert plan_row is not None
            row_status = PlanStatus(plan_row.status)
            if side == ExecutionSide.BUY and row_status in (
                PlanStatus.CANCELLED,
                PlanStatus.EXPIRED,
                PlanStatus.INVALIDATED,
                PlanStatus.REJECTED,
            ):
                deviations.append(f"计划状态为 {row_status.value}")
            if strict_plan and deviations:
                raise PortfolioError("; ".join(deviations))
        elif plan_id and plan_row is None:
            deviations.append("关联的计划不存在，策略字段标记为待补全")
        deviation_text = "；".join(deviations) or None

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
        strategy_version = (
            domain.strategy_version
            if domain is not None
            else (plan_row.strategy_version if plan_row is not None else None)
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
            strategy=strategy.value if strategy else None,
            sub_mode=sub_mode.value if sub_mode else None,
            sector=sector,
            strategy_version=strategy_version,
            plan_deviation=deviation_text,
        )
        if not created:
            return execution
        if deviation_text:
            AuditRepository.log(
                self.session,
                actor="user",
                action="execution_deviation",
                target=ts_code,
                after={"plan_id": plan_id, "deviation": deviation_text},
            )
        if side == ExecutionSide.BUY:
            batch_index, initial_r = self._buy(
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
                strategy_version=strategy_version,
                domain=domain,
                needs_reconciliation=plan_id is None or domain is None,
            )
            execution.batch_index = batch_index
            execution.initial_r = (
                Decimal(str(initial_r)) if initial_r is not None else None
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
        strategy_version: str | None,
        domain: Any,
        needs_reconciliation: bool,
    ) -> tuple[int, Decimal | None]:
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

        explicit_add_on = bool(
            domain is not None and domain.position_plan.is_add_on
        )
        different_plan = bool(
            position is not None
            and position.plan_id
            and plan_id
            and position.plan_id != plan_id
        )
        is_add_on = explicit_add_on or different_plan
        same_plan_lots = [
            lot
            for lot in (position.lots if position is not None else [])
            if plan_id and lot.plan_id == plan_id
        ]
        if is_add_on:
            if same_plan_lots:
                batch_index = same_plan_lots[0].add_index
            else:
                max_index = max(
                    (
                        lot.add_index
                        for lot in (position.lots if position is not None else [])
                    ),
                    default=0,
                )
                batch_index = max(max_index, position.add_on_count if position else 0) + 1
        else:
            batch_index = 0

        initial_stop: Decimal | None = None
        if domain is not None:
            initial_stop = domain.stop_plan.initial_stop
        elif position is not None and position.current_stop > 0:
            initial_stop = position.current_stop
        initial_r = (
            price - initial_stop
            if initial_stop is not None and initial_stop > 0
            else None
        )
        proposed_stop = initial_stop
        if position is not None and position.current_stop > 0:
            if proposed_stop is None:
                proposed_stop = position.current_stop
            else:
                proposed_stop = max(proposed_stop, position.current_stop)

        if position is None:
            position = PositionRepository.upsert(
                self.session,
                account_id=self.account_id,
                ts_code=ts_code,
                sector=sector,
                shares=new_shares,
                available_shares=0,
                cost_price=new_cost,
                current_stop=proposed_stop or Decimal("0"),
                current_price=price,
                primary_strategy=strategy.value if strategy else None,
                sub_mode=sub_mode.value if sub_mode else None,
                strategy_version=strategy_version,
                plan_id=plan_id,
                opened_at=executed_at,
                highest_price=price,
                last_new_high_on=executed_at.date(),
                initial_r=initial_r,
                initial_shares=shares,
                add_on_count=0,
                target_state_json=[],
                needs_reconciliation=needs_reconciliation,
                valuation_status=ValuationStatus.OK.value,
                valuation_at=executed_at,
            )
        else:
            position.shares = new_shares
            position.cost_price = new_cost
            if proposed_stop is not None:
                position.current_stop = proposed_stop
            position.current_price = price
            prior_high = position.highest_price or price
            position.highest_price = max(prior_high, price)
            if price > prior_high + Decimal("1e-9"):
                position.last_new_high_on = executed_at.date()
            if strategy and not position.primary_strategy:
                position.primary_strategy = strategy.value
            if sub_mode and not position.sub_mode:
                position.sub_mode = sub_mode.value
            if strategy_version and not position.strategy_version:
                position.strategy_version = strategy_version
            if not is_add_on:
                position.plan_id = plan_id or position.plan_id
                position.initial_shares = max(0, position.initial_shares) + shares
                if position.initial_r is None and initial_r is not None:
                    position.initial_r = initial_r
            else:
                position.add_on_count = max(
                    position.add_on_count, batch_index
                )
            position.needs_reconciliation = bool(
                position.needs_reconciliation or needs_reconciliation
            )

        self._update_target_state(
            position,
            domain=domain,
            shares=shares,
            initial_shares=position.initial_shares,
            is_add_on=is_add_on,
        )
        lot = PositionLotRow(
            id=uuid4().hex,
            position_id=position.id,
            strategy=strategy.value if strategy else None,
            sub_mode=sub_mode.value if sub_mode else None,
            sector=sector,
            strategy_version=strategy_version,
            shares=shares,
            cost_price=price,
            initial_r=initial_r,
            initial_stop=initial_stop,
            entry_fees=fees,
            is_add_on=is_add_on,
            opened_at=executed_at,
            add_index=batch_index,
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
            after={
                "shares": shares,
                "price": float(price),
                "fees": float(fees),
                "batch_index": batch_index,
                "initial_r": float(initial_r) if initial_r is not None else None,
            },
        )
        return batch_index, initial_r

    def _update_target_state(
        self,
        position: PositionRow,
        *,
        domain: Any,
        shares: int,
        initial_shares: int,
        is_add_on: bool,
    ) -> None:
        if domain is None:
            return
        existing = [
            item
            for item in (position.target_state_json or [])
            if isinstance(item, dict)
        ]
        if not is_add_on and not any(
            int(item.get("filled_shares") or 0) > 0 for item in existing
        ):
            from app.portfolio.exit_rules import target_states_from_plan

            states = target_states_from_plan(
                [target.model_dump(mode="json") for target in domain.take_profit_plan.targets],
                total_shares=position.shares,
                initial_shares=initial_shares,
            )
            position.target_state_json = [
                state.model_dump(mode="json") for state in states
            ]
            return
        if is_add_on:
            from app.portfolio.exit_rules import target_states_from_plan

            additional = target_states_from_plan(
                [target.model_dump(mode="json") for target in domain.take_profit_plan.targets],
                total_shares=position.shares,
                initial_shares=shares,
            )
            merged: dict[float, dict] = {
                float(item.get("price") or 0): dict(item) for item in existing
            }
            for state in additional:
                key = float(state.price)
                if key in merged:
                    merged[key]["planned_shares"] = int(
                        merged[key].get("planned_shares") or 0
                    ) + state.planned_shares
                else:
                    merged[key] = state.model_dump(mode="json")
            position.target_state_json = list(merged.values())

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
        risk_per_share = (
            float(position.initial_r)
            if position.initial_r is not None and position.initial_r > 0
            else None
        )
        if risk_per_share:
            position.realized_r += (
                (float(price) - float(position.cost_price)) / risk_per_share
            ) * shares
        self._allocate_target_fills(position, shares)
        if remaining <= 0:
            # Deleting the position cascades to its lots; reducing them first
            # would make the ORM try to delete the same rows twice.
            pass
        else:
            self._reduce_lots(position, shares)
        if remaining <= 0:
            PositionRepository.delete(self.session, self.account_id, ts_code)
            for row in PlanRepository.list_open(self.session, self.account_id):
                if row.ts_code == ts_code:
                    PlanRepository.update_status(
                        self.session, row.id, PlanStatus.CLOSED, "position closed"
                    )
        else:
            position.shares = remaining
            position.current_price = price
            position.highest_price = max(
                position.highest_price or price, price
            )
            self.refresh_available(executed_at.date())
        AuditRepository.log(
            self.session,
            actor="user",
            action="sell",
            target=ts_code,
            after={
                "shares": shares,
                "price": float(price),
                "fees": float(fees),
                "remaining": max(0, remaining),
            },
        )

    def _allocate_target_fills(self, position: PositionRow, shares: int) -> None:
        """Mark take-profit tranches as filled without deleting their state.

        The allocation is FIFO across the plan's targets.  A stop-out closes
        the position anyway; a partial manual sell is at least visible in the
        tranche progress rather than silently disappearing.
        """
        states = [
            dict(item)
            for item in (position.target_state_json or [])
            if isinstance(item, dict)
        ]
        remaining = int(shares)
        for item in states:
            if remaining <= 0:
                break
            planned = int(item.get("planned_shares") or 0)
            filled = int(item.get("filled_shares") or 0)
            available = max(0, planned - filled)
            if available <= 0:
                item["status"] = "done"
                continue
            take = min(remaining, available)
            item["filled_shares"] = filled + take
            remaining -= take
            item["status"] = (
                "done"
                if item["filled_shares"] >= planned
                else "partial"
            )
        position.target_state_json = states

    def _reduce_lots(self, position: PositionRow, shares: int) -> None:
        remaining = shares
        for lot in PositionRepository.list_lots(self.session, position.id):
            if remaining <= 0:
                break
            reduce = min(remaining, lot.shares)
            lot.shares -= reduce
            remaining -= reduce
            if lot.shares == 0:
                if lot in position.lots:
                    position.lots.remove(lot)
                self.session.delete(lot)
        if remaining > 0:
            raise PortfolioError("lot accounting mismatch")

    def _update_plan_after_buy(self, plan_id: str) -> None:
        plan = PlanRepository.get(self.session, plan_id)
        if plan is None:
            return
        if plan.status not in (
            PlanStatus.PENDING_EXECUTION.value,
            PlanStatus.PENDING_ENTRY.value,
            PlanStatus.PARTIAL_FILL.value,
        ):
            # A real fill against an expired/cancelled plan is still recorded
            # for audit, but it must not resurrect the plan state.
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
        performance_equity = account.equity - (account.net_deposits or Decimal("0"))
        account.peak_equity = max(account.peak_equity, performance_equity)
        account.last_synced_at = datetime.now(UTC)
        account.last_valuation_at = datetime.now(UTC)
        # Deliberately do not touch last_reconciled_at: a fill or a valuation
        # is not a full manual reconciliation of the broker account.
        self.session.flush()

    def revalue(
        self,
        as_of: date,
        points: dict[str, PricePoint],
        *,
        now: datetime | None = None,
    ) -> AccountRow:
        """Unified mark-to-market step.

        Missing/stale quotes keep the last known price, mark the valuation
        incomplete, and never zero the position value.  External cash flows
        are excluded from the peak/drawdown baseline.
        """
        account = self.account()
        incomplete: list[str] = []
        positions = PositionRepository.list_positions(self.session, self.account_id)
        for position in positions:
            point = points.get(position.ts_code)
            if (
                point is not None
                and point.data_time is not None
                and position.opened_at is not None
                and point.data_time.date() < position.opened_at.date()
            ):
                point = None
            if point is None:
                position.valuation_status = ValuationStatus.MISSING.value
                position.valuation_at = now or datetime.now(UTC)
                incomplete.append(f"{position.ts_code}: 缺少行情")
                continue
            if not point.usable:
                position.valuation_status = point.status.value
                position.valuation_at = point.data_time or now or datetime.now(UTC)
                issues = "；".join(point.issues) or point.status.value
                incomplete.append(f"{position.ts_code}: {issues}")
                continue
            position.current_price = Decimal(str(point.price))
            position.valuation_status = ValuationStatus.OK.value
            position.valuation_at = point.data_time or now or datetime.now(UTC)
        value = sum(
            (position.current_price * position.shares for position in positions),
            Decimal("0"),
        )
        account.equity = account.cash + value
        performance_equity = account.equity - (account.net_deposits or Decimal("0"))
        account.peak_equity = max(account.peak_equity, performance_equity)
        account.valuation_incomplete = bool(incomplete)
        account.valuation_note = "；".join(incomplete) or None
        account.last_valuation_at = now or datetime.now(UTC)
        account.last_synced_at = account.last_valuation_at
        self.session.flush()
        return account

    def reconcile(
        self,
        *,
        equity: Decimal | float | None = None,
        cash: Decimal | float | None = None,
        note: str | None = None,
        reconciled_at: datetime | None = None,
    ) -> AccountRow:
        """Record an explicit human reconciliation of broker balances."""
        account = self.account()
        if equity is not None:
            account.equity = Decimal(str(equity))
        if cash is not None:
            account.cash = Decimal(str(cash))
        account.last_reconciled_at = reconciled_at or datetime.now(UTC)
        account.stale = False
        account.risk_profile_json = {
            **(account.risk_profile_json or {}),
            "last_reconciliation_note": note,
        }
        AuditRepository.log(
            self.session,
            actor="user",
            action="reconcile_account",
            target=self.account_id,
            after={
                "equity": float(account.equity),
                "cash": float(account.cash),
                "note": note,
            },
        )
        self._recompute_equity(account)
        return account

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
        value = Decimal(str(amount))
        if flow_type in ("withdraw", "withdrawal") and value > 0:
            value = -value
        _, created = CashFlowRepository.add(
            self.session,
            account_id=self.account_id,
            flow_type=flow_type,
            amount=value,
            occurred_at=occurred_at,
            external_id=external_id,
            note=note,
        )
        if created:
            account.cash += value
            if flow_type in ("deposit", "withdraw", "withdrawal"):
                account.net_deposits = (
                    account.net_deposits or Decimal("0")
                ) + value
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
        if share_ratio and share_ratio != -1:
            ratio = Decimal(str(1 + share_ratio))
            new_shares = int(position.shares * (1 + share_ratio) // 100 * 100)
            if new_shares > 0:
                position.cost_price = position.cost_price * position.shares / new_shares
                if position.current_stop > 0:
                    position.current_stop = position.current_stop / ratio
                if position.initial_r is not None:
                    position.initial_r = position.initial_r / ratio
                if position.highest_price is not None:
                    position.highest_price = position.highest_price / ratio
                position.initial_shares = int(
                    position.initial_shares * (1 + share_ratio)
                )
                position.available_shares = int(
                    position.available_shares * (1 + share_ratio)
                )
                position.shares = new_shares
                lots = PositionRepository.list_lots(self.session, position.id)
                for lot in lots:
                    if lot.cost_price > 0:
                        lot.cost_price = lot.cost_price / ratio
                    if lot.initial_stop is not None:
                        lot.initial_stop = lot.initial_stop / ratio
                    if lot.initial_r is not None:
                        lot.initial_r = lot.initial_r / ratio
                    lot.shares = int(lot.shares * (1 + share_ratio))
                if lots:
                    lots[0].shares += new_shares - sum(
                        lot.shares for lot in lots
                    )
                states = []
                for item in position.target_state_json or []:
                    entry = dict(item)
                    entry["price"] = float(
                        Decimal(str(entry.get("price") or 0)) / ratio
                    )
                    entry["planned_shares"] = int(
                        int(entry.get("planned_shares") or 0) * (1 + share_ratio)
                    )
                    entry["filled_shares"] = int(
                        int(entry.get("filled_shares") or 0) * (1 + share_ratio)
                    )
                    states.append(entry)
                position.target_state_json = states
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
        """Mark a real account stale based on *manual reconciliation* age.

        A nightly valuation updates ``last_valuation_at`` but never claims the
        broker balances were reconciled.
        """
        account = self.account()
        if account.mode == AccountMode.REAL.value:
            reference = (
                account.last_reconciled_at.date()
                if account.last_reconciled_at is not None
                else None
            )
            account.stale = reference is None or reference < as_of
        else:
            account.stale = False
        self.session.flush()
        return account.stale

    def state(self, current_prices: dict[str, Decimal] | None = None) -> AccountState:
        return AccountRepository.to_state(
            self.session, self.account_id, current_prices=current_prices
        )


ACTIVE_ORDER_STATUSES = (
    OrderStatus.PENDING.value,
    OrderStatus.PARTIAL.value,
    OrderStatus.CANCEL_REQUESTED.value,
)


def cancel_order(
    session: Session,
    order_id: str,
    reason: str,
    *,
    confirmed: bool = False,
) -> OrderRow:
    """Request or confirm an order cancellation.

    ``cancel_order`` without confirmation records a broker request that may
    still fill; the order remains in ``cancel_requested`` and keeps consuming
    cash/risk capacity.  Only an explicit broker confirmation releases it.
    """
    order = session.get(OrderRow, order_id)
    if order is None:
        raise PortfolioError("order not found")
    if order.status == OrderStatus.FILLED.value:
        raise PortfolioError("cannot cancel a filled order")
    if order.status in (
        OrderStatus.CANCELLED.value,
        OrderStatus.REJECTED.value,
    ):
        return order
    order.status = (
        OrderStatus.CANCELLED.value
        if confirmed
        else OrderStatus.CANCEL_REQUESTED.value
    )
    order.cancel_reason = reason
    AuditRepository.log(
        session,
        actor="user",
        action="confirm_cancel_order" if confirmed else "request_cancel_order",
        target=order_id,
        after={"reason": reason, "status": order.status},
    )
    session.flush()
    return order


def confirm_order_cancel(
    session: Session, order_id: str, reason: str = "broker confirmed"
) -> OrderRow:
    order = session.get(OrderRow, order_id)
    if order is None:
        raise PortfolioError("order not found")
    if order.status == OrderStatus.FILLED.value:
        raise PortfolioError("cannot cancel a filled order")
    return cancel_order(session, order_id, reason, confirmed=True)


def reject_order_cancel(
    session: Session, order_id: str, reason: str = "broker rejected cancel"
) -> OrderRow:
    """The broker refused the cancel: revert to the actual working state."""
    order = session.get(OrderRow, order_id)
    if order is None:
        raise PortfolioError("order not found")
    if order.status not in ACTIVE_ORDER_STATUSES:
        return order
    filled = int(order.filled_shares or 0)
    order.status = (
        OrderStatus.PARTIAL.value if filled > 0 else OrderStatus.PENDING.value
    )
    order.cancel_reason = reason
    AuditRepository.log(
        session,
        actor="user",
        action="reject_cancel_order",
        target=order_id,
        after={"reason": reason, "status": order.status},
    )
    session.flush()
    return order


def record_order_fill(
    session: Session,
    order_id: str,
    *,
    filled_shares: int | None = None,
    add_shares: int | None = None,
    reason: str | None = None,
) -> OrderRow:
    """Apply a broker fill fact without inventing an execution record.

    ``filled_shares`` is the cumulative filled quantity when supplied;
    ``add_shares`` is an increment.  Remaining exposure is recalculated from
    this value, never from the original order quantity.
    """
    order = session.get(OrderRow, order_id)
    if order is None:
        raise PortfolioError("order not found")
    if order.status in (
        OrderStatus.CANCELLED.value,
        OrderStatus.REJECTED.value,
    ):
        raise PortfolioError("cannot fill a cancelled/rejected order")
    if filled_shares is not None:
        filled = int(filled_shares)
    elif add_shares is not None:
        filled = int(order.filled_shares or 0) + int(add_shares)
    else:
        raise PortfolioError("filled_shares or add_shares is required")
    if filled < 0 or filled > order.shares:
        raise PortfolioError("filled quantity is out of range")
    order.filled_shares = filled
    order.status = (
        OrderStatus.FILLED.value
        if filled >= order.shares
        else OrderStatus.PARTIAL.value
        if filled > 0
        else OrderStatus.PENDING.value
    )
    if reason:
        order.cancel_reason = reason
    AuditRepository.log(
        session,
        actor="user",
        action="order_fill",
        target=order_id,
        after={"filled_shares": filled, "status": order.status},
    )
    session.flush()
    return order


def cancel_plan(
    session: Session, plan_id: str, reason: str = "user cancelled"
) -> dict:
    """Cancel the local plan without pretending broker orders are cancelled.

    The plan status changes, but any pending/partial/cancel-requested order
    keeps its remaining quantity and therefore its cash/risk reservation.
    """
    plan = session.get(PlanRow, plan_id)
    if plan is None:
        raise PortfolioError("plan not found")
    orders = list(
        session.scalars(
            select(OrderRow).where(
                OrderRow.plan_id == plan_id,
                OrderRow.status.in_(list(ACTIVE_ORDER_STATUSES)),
            )
        )
    )
    remaining = sum(
        max(0, int(order.shares) - int(order.filled_shares or 0))
        for order in orders
    )
    PlanRepository.update_status(
        session, plan_id, PlanStatus.CANCELLED, reason
    )
    AuditRepository.log(
        session,
        actor="user",
        action="cancel_plan",
        target=plan_id,
        after={
            "status": PlanStatus.CANCELLED.value,
            "active_orders": [order.id for order in orders],
            "remaining_shares": remaining,
        },
    )
    return {
        "plan_id": plan_id,
        "status": PlanStatus.CANCELLED.value,
        "active_orders": [order.id for order in orders],
        "remaining_shares": remaining,
        "exposure_retained": remaining > 0,
        "message": (
            "计划已取消，但仍有未确认撤销的委托；资金/风险占用保留，"
            "需在券商端撤单并确认。"
            if remaining > 0
            else "计划已取消，无活动委托。"
        ),
    }
