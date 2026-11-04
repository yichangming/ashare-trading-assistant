"""Daily HTML/Markdown report generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.full_config import BACKEND_ROOT
from app.domain.enums import Strategy
from app.domain.models import AccountState, PositionDecision
from app.jobs.report_text import (
    action_label,
    breadth_label,
    breadth_note,
    candidate_stage_label,
    decision_source_label,
    degraded_note,
    degraded_short,
    format_datetime,
    global_risk_label,
    global_risk_note,
    global_risk_reason,
    liquidity_label,
    liquidity_note,
    markdown_to_html,
    order_status_label,
    permission_label,
    permission_note,
    permission_reason,
    plan_status_label,
    reason_label,
    rejection_sentence,
    sentiment_label,
    sentiment_note,
    stale_reason_label,
    strategy_label,
    sub_mode_label,
    target_status_label,
    trend_label,
    trend_note,
    valuation_label,
    volatility_label,
    volatility_note,
)
from app.pipeline_v2 import StrategyPipelineResult


@dataclass
class _Funnel:
    candidates: int = 0
    observing: int = 0
    confirmed: int = 0
    rejected: list[str] = field(default_factory=list)


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _percent(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return str(value)


class ReportService:
    def __init__(self, reports_root: Path | None = None):
        self.reports_root = reports_root or BACKEND_ROOT / "data" / "reports"

    def generate(
        self,
        result: StrategyPipelineResult | None,
        *,
        as_of=None,
        account: AccountState | None = None,
        decisions: list[PositionDecision] | None = None,
        incomplete: list[str] | None = None,
        data_notes: list[str] | None = None,
        position_reviewed: bool = False,
    ) -> dict[str, str]:
        day_value = as_of or (result.as_of if result is not None else None)
        if day_value is None:
            raise ValueError("as_of or result is required")
        day = day_value.isoformat()
        decisions = decisions or []
        incomplete = incomplete or []
        data_notes = data_notes or []
        state = account or (result.account if result is not None else None)
        directory = self.reports_root / day
        directory.mkdir(parents=True, exist_ok=True)
        markdown = self._markdown(
            day,
            result,
            state,
            decisions,
            incomplete,
            data_notes,
            position_reviewed,
        )
        html_text = self._html(day, markdown)
        md_path = directory / "daily.md"
        html_path = directory / "daily.html"
        md_path.write_text(markdown, encoding="utf-8")
        html_path.write_text(html_text, encoding="utf-8")
        return {
            "markdown": markdown,
            "html": html_text,
            "markdown_path": str(md_path),
            "html_path": str(html_path),
        }

    def _markdown(
        self,
        day: str,
        result: StrategyPipelineResult | None,
        account: AccountState | None,
        decisions: list[PositionDecision],
        incomplete: list[str],
        data_notes: list[str],
        position_reviewed: bool,
    ) -> str:
        lines = [
            f"# A 股交易辅助日报 — {day}",
            "",
            "> 研究/决策辅助，不构成投资建议；所有成交由人工执行。",
            "",
            "## 今日摘要",
            "",
            *[
                f"- {item}"
                for item in self._summary_lines(
                    result, decisions, incomplete, position_reviewed
                )
            ],
            "",
        ]
        if incomplete or data_notes:
            lines.extend(["## 数据完整性", ""])
            if incomplete:
                lines.extend(
                    [
                        "本次运行标记为数据不完整，已停止新增信号；"
                        "已有持仓保护仍按规则运行：",
                        "",
                        *[f"- {item}" for item in incomplete],
                        "",
                    ]
                )
            if data_notes:
                lines.extend(
                    [
                        "补充数据说明：",
                        "",
                        *[f"- {item}" for item in data_notes],
                        "",
                    ]
                )
        if result is not None:
            lines.extend(["## 市场环境", ""])
            lines.extend(self._market_table(result))
            lines.append("")
            lines.extend(
                [
                    "## 分策略权限",
                    "",
                    *self._permission_table(result),
                    "",
                    "> 风险系数是策略风险预算的乘数，不是持仓比例："
                    "1.00 表示按正常风险预算；0.50 表示按正常预算的 50%；"
                    "0.00 表示不分配新风险预算、不生成新计划。",
                    "",
                ]
            )
            lines.extend(["## 策略漏斗", ""])
            lines.extend(self._funnel_table(result))
            lines.extend(["", "### 计划拒绝原因", ""])
            rejection_lines = self._rejection_sentences(result)
            if rejection_lines:
                lines.extend(f"- {item}" for item in rejection_lines)
            else:
                lines.append("无计划拒绝。")
            no_candidate_lines = self._no_candidate_lines(result)
            if no_candidate_lines:
                lines.extend(["", "### 无候选说明", ""])
                lines.extend(f"- {item}" for item in no_candidate_lines)
            lines.extend(["", "## 候选与计划", ""])
            lines.extend(self._candidate_table(result))
            lines.append("")
            lines.extend(self._plan_table(result))

        lines.extend(["", "## 持仓行动建议", ""])
        if not decisions:
            if position_reviewed:
                lines.append("持仓监控已运行，当前账户无持仓。")
            else:
                lines.append(
                    "持仓监控未运行或缺少持仓结果，无法判断持仓操作；"
                    "请检查每日任务状态和持仓数据。"
                )
        for decision in decisions:
            lines.extend(self._decision_lines(decision))
            lines.append("")

        lines.extend(["## 账户", ""])
        lines.extend(self._account_table(account))
        lines.extend(
            [
                "",
                "## 限制",
                "",
                "- B2 盘中模式需要 5 分钟数据；日报只反映日线盘后结果。",
                "- 数据/事件不足时停止新增信号并标注不完整，不伪造完整结果。",
                "- 提醒/计划不等于成交；只有录入成交后才改变现金与持仓。",
            ]
        )
        return "\n".join(lines)

    # ---- summary ------------------------------------------------------
    def _summary_lines(
        self,
        result: StrategyPipelineResult | None,
        decisions: list[PositionDecision],
        incomplete: list[str],
        position_reviewed: bool,
    ) -> list[str]:
        if result is None:
            market = (
                "**市场状态**：当日策略分析未完成，无法给出完整市场判断。"
            )
        else:
            regime = result.regime
            degraded = "、".join(
                degraded_short(flag) for flag in regime.degraded_flags
            ) or "无"
            market = (
                f"**市场状态**：{trend_label(regime.trend_regime)}；"
                f"情绪阶段：{sentiment_label(regime.sentiment_stage)}；"
                f"市场宽度：{_percent(regime.indicators.breadth)}"
                f"（{breadth_label(regime.breadth_label)}）；"
                f"市场风险系数：{regime.risk_factor:.2f}"
                "（风险预算乘数，不是持仓比例）；"
                f"全局风险开关：{global_risk_label(regime.global_risk)}；"
                f"数据降级：{degraded}。"
            )
        if result is None:
            plan_line = "**交易计划**：当日未完成选股分析，没有生成新交易计划。"
        elif result.plans:
            plan_codes = "、".join(
                f"{strategy_label(plan.strategy)} {plan.candidate.ts_code}"
                for plan in result.plans
            )
            plan_line = (
                f"**交易计划**：生成 {len(result.plans)} 个待执行计划"
                f"（{plan_codes}）。"
            )
            if result.rejected:
                plan_line += (
                    f"另有 {len(result.rejected)} 只确认候选未生成计划，"
                    "详见策略漏斗。"
                )
        else:
            rejection_lines = self._rejection_sentences(result)
            if rejection_lines:
                plan_line = (
                    "**交易计划**：未生成新交易计划；"
                    + " ".join(rejection_lines)
                )
            else:
                plan_line = (
                    "**交易计划**：未生成新交易计划；"
                    "当日没有通过确认并满足风控的候选。"
                )
        if position_reviewed:
            if not decisions:
                position_line = (
                    "**持仓提醒**：持仓监控已运行；当前账户无持仓，"
                    "无需操作提醒。"
                )
            else:
                exits = sum(
                    1 for item in decisions if item.action.value == "exit"
                )
                reduces = sum(
                    1 for item in decisions if item.action.value == "reduce"
                )
                anomalies = sum(
                    1
                    for item in decisions
                    if item.action.value == "data_anomaly"
                )
                waits = sum(
                    1
                    for item in decisions
                    if item.action.value == "wait_execution"
                )
                risk_count = exits + reduces + anomalies + waits
                if risk_count:
                    parts = []
                    if exits:
                        parts.append(f"清仓 {exits}")
                    if reduces:
                        parts.append(f"减仓 {reduces}")
                    if anomalies:
                        parts.append(f"数据异常 {anomalies}")
                    if waits:
                        parts.append(f"等待委托 {waits}")
                    position_line = (
                        "**持仓提醒**：持仓监控已运行；"
                        f"{risk_count} 只持仓需要处理"
                        f"（{'、'.join(parts)}）。"
                    )
                else:
                    adds = sum(
                        1
                        for item in decisions
                        if item.action.value == "conditional_add"
                    )
                    position_line = (
                        "**持仓提醒**：持仓监控已运行；"
                        f"{len(decisions)} 只持仓均无操作提醒。"
                    )
                    if adds:
                        position_line = (
                            "**持仓提醒**：持仓监控已运行；"
                            f"{len(decisions)} 只持仓无强制操作提醒，"
                            f"其中 {adds} 只可继续观察条件加仓。"
                        )
        else:
            position_line = (
                "**持仓提醒**：持仓监控未运行或缺少持仓结果，"
                "无法判断是否需要操作；请检查每日任务和持仓数据。"
            )
        if incomplete:
            data_line = (
                "**数据完整性**：存在不完整项，已停止新增信号；"
                "具体原因见下方数据完整性说明。"
            )
        elif result is None:
            data_line = "**数据完整性**：当日没有可用的策略分析结果。"
        else:
            data_line = (
                "**数据完整性**：数据更新和质量检查未标记不完整。"
            )
        return [market, plan_line, position_line, data_line]

    # ---- market and permissions --------------------------------------
    def _market_table(self, result: StrategyPipelineResult) -> list[str]:
        regime = result.regime
        degraded = "、".join(
            degraded_short(flag) for flag in regime.degraded_flags
        ) or "无"
        degraded_notes = "；".join(
            degraded_note(flag) for flag in regime.degraded_flags
        ) or "数据未标记降级。"
        lines = [
            "| 项目 | 当前值 | 说明 |",
            "|---|---|---|",
            f"| 趋势 | {trend_label(regime.trend_regime)} | "
            f"{trend_note(regime.trend_regime)} |",
            f"| 情绪阶段 | {sentiment_label(regime.sentiment_stage)} | "
            f"{sentiment_note(regime.sentiment_stage)} |",
            f"| 市场风险系数 | {regime.risk_factor:.2f} | "
            "市场层风险预算乘数，不是账户持仓比例。 |",
            f"| 市场宽度 | {_percent(regime.indicators.breadth)}"
            f"（{breadth_label(regime.breadth_label)}） | "
            f"{breadth_note(regime.breadth_label)} |",
            f"| 波动 | {volatility_label(regime.volatility_label)} | "
            f"{volatility_note(regime.volatility_label)} |",
            f"| 流动性 | {liquidity_label(regime.liquidity_label)} | "
            f"{liquidity_note(regime.liquidity_label)} |",
            f"| 全局风险开关 | {global_risk_label(regime.global_risk)} | "
            f"{global_risk_note(regime.global_risk)} |",
            f"| 数据降级 | {degraded} | {degraded_notes} |",
        ]
        if regime.risk_reasons:
            lines.append(
                "| 全局风险原因 | "
                + "；".join(global_risk_reason(item) for item in regime.risk_reasons)
                + " | 触发全局风险开关的原因。 |"
            )
        return lines

    def _permission_table(self, result: StrategyPipelineResult) -> list[str]:
        lines = [
            "| 策略 | 权限 | 风险系数 | 说明 |",
            "|---|---|---|---|",
        ]
        for strategy in Strategy:
            permission = result.regime.strategy_permissions.get(strategy)
            if permission is None:
                lines.append(
                    f"| {strategy_label(strategy)} | 未知 | — | "
                    "未提供权限结果。 |"
                )
                continue
            reasons = "；".join(
                permission_reason(item) for item in permission.reasons
            )
            note = permission_note(permission.mode)
            if reasons:
                note = f"{note} 权限原因：{reasons}。"
            lines.append(
                f"| {strategy_label(strategy)} | "
                f"{permission_label(permission.mode)} | "
                f"{permission.risk_factor:.2f} | {note} |"
            )
        return lines

    # ---- funnel and rejections ---------------------------------------
    def _funnel(self, result: StrategyPipelineResult) -> dict[str, _Funnel]:
        funnel: dict[str, _Funnel] = defaultdict(_Funnel)
        for candidate in result.candidates:
            key = candidate.strategy.value if candidate.strategy else "?"
            entry = funnel[key]
            entry.candidates += 1
            stage = str(candidate.evidence.get("stage", "observing"))
            if stage == "confirmed":
                entry.confirmed += 1
            else:
                entry.observing += 1
        for rejected in result.rejected:
            key = (
                rejected.candidate.strategy.value
                if rejected.candidate.strategy
                else "?"
            )
            funnel[key].rejected.append(
                f"{rejected.candidate.ts_code}: {rejected.reason}"
            )
        return funnel

    def _funnel_table(self, result: StrategyPipelineResult) -> list[str]:
        funnel = self._funnel(result)
        lines = [
            "| 策略 | 权限 | 候选 | 观察 | 确认 | 说明 |",
            "|---|---|---|---|---|---|",
        ]
        for strategy in Strategy:
            permission = result.regime.strategy_permissions.get(strategy)
            mode = permission.mode if permission is not None else None
            entry = funnel[strategy.value]
            notes: list[str] = []
            if permission is not None and permission.reasons:
                notes.extend(
                    permission_reason(item) for item in permission.reasons
                )
            if entry.rejected:
                notes.append(f"{len(entry.rejected)} 只确认候选未生成计划")
            if not notes:
                notes.append("无")
            lines.append(
                f"| {strategy_label(strategy)} | "
                f"{permission_label(mode) if mode is not None else '未知'} | "
                f"{entry.candidates} | {entry.observing} | "
                f"{entry.confirmed} | {'；'.join(notes)} |"
            )
        return lines

    def _rejection_sentences(
        self, result: StrategyPipelineResult
    ) -> list[str]:
        plan_codes = {plan.candidate.ts_code for plan in result.plans}
        sentences: list[str] = []
        for strategy in Strategy:
            rejected = [
                (item.candidate.ts_code, item.reason)
                for item in result.rejected
                if item.candidate.strategy == strategy
            ]
            if not rejected:
                continue
            confirmed = sum(
                1
                for candidate in result.candidates
                if candidate.strategy == strategy
                and candidate.evidence.get("stage") == "confirmed"
            )
            confirmed = max(confirmed, len(rejected))
            sentences.append(
                rejection_sentence(
                    strategy, confirmed, rejected, plan_codes=plan_codes
                )
            )
        return sentences

    def _no_candidate_lines(self, result: StrategyPipelineResult) -> list[str]:
        funnel = self._funnel(result)
        lines: list[str] = []
        for strategy in Strategy:
            permission = result.regime.strategy_permissions.get(strategy)
            if permission is not None and permission.mode.value == "paused":
                continue
            entry = funnel[strategy.value]
            if entry.candidates:
                continue
            if strategy == Strategy.F:
                lines.append(
                    f"{strategy_label(strategy)}：需要先有 C1/C2 形态候选，"
                    "并有已核验、方向正面的对应事件；本次没有同时满足的标的，"
                    "未降低事件标准。"
                )
            else:
                lines.append(
                    f"{strategy_label(strategy)}：按该策略的价格、形态和"
                    "数据门槛过滤后没有候选，未降低标准强行选股。"
                )
        return lines

    # ---- candidates, plans and decisions -----------------------------
    def _candidate_table(self, result: StrategyPipelineResult) -> list[str]:
        if not result.candidates:
            return ["当日无候选。"]
        lines = [
            "| 策略 | 子模式 | 代码 | 行业 | 评分 | 阶段 |",
            "|---|---|---|---|---|---|",
        ]
        for candidate in result.candidates:
            stage = str(candidate.evidence.get("stage", "observing"))
            lines.append(
                f"| {strategy_label(candidate.strategy)} | "
                f"{sub_mode_label(candidate.sub_mode.value if candidate.sub_mode else None)} | "
                f"{candidate.ts_code} | {candidate.sector} | "
                f"{candidate.score:.1f} | {candidate_stage_label(stage)} |"
            )
        return lines

    def _plan_table(self, result: StrategyPipelineResult) -> list[str]:
        if not result.plans:
            return ["当日未生成新交易计划。"]
        lines = [
            "| 策略 | 代码 | 状态 | 拟入场价 | 止损价 | 计划股数 |",
            "|---|---|---|---|---|---|",
        ]
        for plan in result.plans:
            lines.append(
                f"| {strategy_label(plan.strategy)} | "
                f"{plan.candidate.ts_code} | "
                f"{plan_status_label(plan.status)} | "
                f"{plan.entry_plan.entry_price} | "
                f"{plan.stop_plan.initial_stop} | "
                f"{plan.position_plan.shares} |"
            )
        return lines

    def _decision_lines(self, decision: PositionDecision) -> list[str]:
        strategy = strategy_label(decision.primary_strategy)
        sub_mode = sub_mode_label(
            decision.sub_mode.value if decision.sub_mode else None
        )
        targets = "；".join(
            f"{target.price} 元（{target_status_label(target.status)}）"
            for target in decision.targets
        ) or "无"
        lines = [
            f"### {decision.ts_code} · {strategy} / {sub_mode}",
            f"- 动作：**{action_label(decision.action)}**"
            f"（原因：{reason_label(decision.reason_code)}）",
            f"- 说明：{decision.explanation}",
            f"- 持仓：总 {decision.total_shares} 股，可卖 "
            f"{decision.available_shares} 股，建议 "
            f"{decision.suggested_shares} 股，受限 "
            f"{decision.restricted_shares} 股，委托覆盖 "
            f"{decision.covered_shares} 股。",
            f"- 保护线：{decision.current_stop or '待补全'}；"
            f"待生效：{decision.pending_stop or '无'}"
            + (
                f"（生效时间：{format_datetime(decision.pending_stop_effective_at)}）"
                if decision.pending_stop_effective_at
                else ""
            )
            + f"；目标：{targets}。",
            f"- 数据：{format_datetime(decision.data_time)}；"
            f"来源：{decision_source_label(decision.decision_source)}；"
            f"估值状态：{valuation_label(decision.valuation_status)}；"
            f"当前有效：{'是' if decision.is_current else '否'}。",
            f"- 成交：已成交 {decision.executed_shares} 股；"
            f"待成交 {decision.pending_shares} 股。",
        ]
        if decision.stale_reason:
            lines.append(
                f"- 过期原因：{stale_reason_label(decision.stale_reason)}。"
            )
        for order in decision.pending_sell_orders:
            lines.append(
                f"- 未成交卖单 {order.get('order_id')}："
                f"价格 {order.get('price')}，"
                f"状态 {order_status_label(order.get('status'))}，"
                f"原始 {order.get('shares')} 股，已成交 "
                f"{order.get('filled_shares')} 股，剩余 "
                f"{order.get('remaining_shares')} 股。"
            )
        if decision.days_since_new_high is not None:
            lines.append(
                f"- 最近新高：{decision.last_new_high_on or '未知'}；"
                f"距最近新高 {decision.days_since_new_high} 个交易日。"
            )
        lines.append(f"- 下一观察条件：{self._next_condition(decision)}")
        return lines

    @staticmethod
    def _next_condition(decision: PositionDecision) -> str:
        if decision.action.value == "data_anomaly":
            return "补全行情、保护线或初始 R 后重新评估。"
        if decision.action.value == "exit":
            return "人工执行后录入成交；未成交前风险仍占用。"
        if decision.action.value == "reduce":
            return "分批止盈执行后核对剩余仓位与保护线。"
        if decision.action.value == "wait_execution":
            return "等待卖出委托成交或撤单；成交前不重复建议。"
        if decision.targets:
            pending = [
                str(target.price)
                for target in decision.targets
                if target.status != "done"
            ]
            if pending:
                return f"观察是否触及目标 {', '.join(pending)} 或跌破保护线。"
        return (
            f"观察是否跌破保护线 {decision.current_stop or '待补全'}，"
            "以及是否达到时间退出条件。"
        )

    # ---- account ------------------------------------------------------
    @staticmethod
    def _account_table(account: AccountState | None) -> list[str]:
        if account is None:
            return ["账户数据未提供。"]
        incomplete = "是" if account.valuation_incomplete else "否"
        if account.valuation_note:
            incomplete += f"（{account.valuation_note}）"
        return [
            "| 项目 | 数值 |",
            "|---|---|",
            f"| 权益 | {_money(account.equity)} |",
            f"| 现金 | {_money(account.cash)} |",
            f"| 净入金（不计收益） | {_money(account.net_deposits)} |",
            f"| 持仓数 | {len(account.positions)} |",
            f"| 待成交计划 | {len(account.pending_plans)} |",
            f"| 估值不完整 | {incomplete} |",
            f"| 最近估值 | {format_datetime(account.last_valuation_at)} |",
            f"| 最近人工核对 | {format_datetime(account.last_reconciled_at)} |",
        ]

    # ---- standalone HTML ---------------------------------------------
    def _html(self, day: str, markdown: str) -> str:
        body = markdown_to_html(markdown)
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>交易辅助日报 {day}</title>
<style>
:root{{color-scheme:light;--bg:#f5f1e8;--text:#1d2b27;--muted:#5b6660;
--line:#d8d0c0;--head:#ece5d6;--accent:#8a6a36}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);
font-family:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
font-size:15px;line-height:1.65}}
main{{max-width:1000px;margin:0 auto;padding:24px 18px 60px}}
h1,h2,h3{{font-family:"Noto Serif SC","Songti SC",serif;line-height:1.35;
margin:1.3em 0 .5em}}
h1{{font-size:1.65rem;margin-top:0}} h2{{font-size:1.25rem}}
h3{{font-size:1.05rem}}
p{{margin:.45rem 0}} ul,ol{{margin:.4rem 0 .8rem;padding-left:1.4rem}}
li{{margin:.2rem 0}}
blockquote{{margin:1rem 0;padding:.2rem 0 .2rem 1rem;
border-left:4px solid var(--accent);color:var(--muted)}}
.table-wrap{{overflow-x:auto;margin:.8rem 0 1.2rem}}
table{{border-collapse:collapse;width:100%;min-width:560px;font-size:.94rem}}
th,td{{border:1px solid var(--line);padding:.5rem .6rem;text-align:left;
vertical-align:top}}
th{{background:var(--head);white-space:nowrap}}
code{{font-family:ui-monospace,Consolas,monospace;background:#e9e2d4;
padding:.08rem .25rem;border-radius:4px}}
hr{{border:0;border-top:1px solid var(--line);margin:1.4rem 0}}
@media (max-width:700px){{
main{{padding:14px 10px 42px}} h1{{font-size:1.35rem}}
h2{{font-size:1.1rem}} table{{min-width:520px;font-size:.88rem}}
th,td{{padding:.4rem .45rem}}
}}
</style>
</head><body><main class="report">{body}</main></body></html>"""
