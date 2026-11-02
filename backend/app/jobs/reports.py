"""Daily HTML/Markdown report generation."""

from __future__ import annotations

import html
from pathlib import Path

from app.core.full_config import BACKEND_ROOT
from app.domain.models import AccountState, PositionDecision
from app.pipeline_v2 import StrategyPipelineResult


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
            day, result, state, decisions, incomplete, data_notes
        )
        html_text = self._html(
            day, result, state, decisions, incomplete, data_notes, markdown
        )
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
    ) -> str:
        lines = [
            f"# A 股交易辅助日报 — {day}",
            "",
            "> 研究/决策辅助，不构成投资建议；所有成交由人工执行。",
            "",
        ]
        if incomplete:
            lines.extend(
                [
                    "## 数据不完整",
                    "",
                    "本次运行未生成新增信号，已有持仓保护仍已执行：",
                    "",
                    *[f"- {item}" for item in incomplete],
                    "",
                ]
            )
        if data_notes:
            lines.extend(
                [
                    "## 数据说明",
                    "",
                    *[f"- {item}" for item in data_notes],
                    "",
                ]
            )
        if result is not None:
            regime = result.regime
            lines.extend(
                [
                    "## 市场环境",
                    "",
                    f"- 趋势：{regime.trend_regime.value}",
                    f"- 情绪阶段：{regime.sentiment_stage.value}",
                    f"- 风险系数：{regime.risk_factor:.2f}",
                    f"- 宽度：{regime.indicators.breadth:.2%}（{regime.breadth_label}）",
                    f"- 波动：{regime.volatility_label}；流动性：{regime.liquidity_label}",
                    f"- 全局风险开关：{regime.global_risk.value}",
                    f"- 数据降级：{', '.join(regime.degraded_flags) or '无'}",
                    "",
                    "## 分策略权限",
                    "",
                ]
            )
            for strategy, permission in regime.strategy_permissions.items():
                lines.append(
                    f"- {strategy.value}：{permission.mode.value}"
                    f"（风险系数 {permission.risk_factor:.2f}）"
                    f"{'；' + '；'.join(permission.reasons) if permission.reasons else ''}"
                )
            lines.extend(["", "## 候选与计划", ""])
            if not result.candidates:
                lines.append("无候选。")
            for candidate in result.candidates:
                lines.append(
                    f"- [{candidate.strategy.value if candidate.strategy else '?'}]"
                    f" {candidate.ts_code} {candidate.sector} "
                    f"score={candidate.score:.1f} stage="
                    f"{candidate.evidence.get('stage', '-')}"
                )
            if result.plans:
                lines.extend(["", "### 计划", ""])
                for plan in result.plans:
                    lines.append(
                        f"- {plan.candidate.ts_code} "
                        f"{plan.strategy.value if plan.strategy else ''}"
                        f" {plan.status.value} 入场={plan.entry_plan.entry_price}"
                        f" 止损={plan.stop_plan.initial_stop}"
                        f" 股数={plan.position_plan.shares}"
                    )

        lines.extend(["", "## 持仓行动建议", ""])
        if not decisions:
            lines.append("无持仓。")
        for decision in decisions:
            strategy_label = (
                decision.primary_strategy.value
                if decision.primary_strategy
                else "待补全"
            )
            sub_mode = decision.sub_mode.value if decision.sub_mode else "-"
            targets = "、".join(
                f"{target.price}({target.status})" for target in decision.targets
            ) or "无"
            lines.append(
                f"### {decision.ts_code} · {strategy_label}/{sub_mode}"
            )
            lines.append(
                f"- 动作：**{decision.action.value}** "
                f"（{decision.reason_code}）"
            )
            lines.append(f"- 原因：{decision.explanation}")
            lines.append(
                f"- 持仓：总 {decision.total_shares} 股，可卖 "
                f"{decision.available_shares} 股，建议 "
                f"{decision.suggested_shares} 股，受限 "
                f"{decision.restricted_shares} 股"
            )
            lines.append(
                f"- 保护线：{decision.current_stop or '待补全'}；"
                f"建议上移：{decision.proposed_stop or '无'}；目标：{targets}"
            )
            lines.append(
                f"- 数据时间：{decision.data_time or '未知'}；"
                f"估值状态：{decision.valuation_status.value}；"
                f"已成交：{decision.executed_shares} 股；"
                f"待成交：{decision.pending_shares} 股"
            )
            lines.append(
                f"- 下一观察条件：{self._next_condition(decision)}"
            )
            lines.append("")

        lines.extend(["## 账户", ""])
        if account is not None:
            lines.extend(
                [
                    f"- 权益：{account.equity}",
                    f"- 现金：{account.cash}",
                    f"- 净入金（不计收益）：{account.net_deposits}",
                    f"- 持仓数：{len(account.positions)}",
                    f"- 待成交计划：{len(account.pending_plans)}",
                    f"- 估值不完整：{'是' if account.valuation_incomplete else '否'}"
                    + (
                        f"（{account.valuation_note}）"
                        if account.valuation_note
                        else ""
                    ),
                    f"- 最近估值：{account.last_valuation_at or '未知'}",
                    f"- 最近人工核对：{account.last_reconciled_at or '未核对'}",
                    "",
                ]
            )
        lines.extend(
            [
                "## 限制",
                "",
                "- B2 盘中模式需要 5 分钟数据；日报只反映日线盘后结果。",
                "- 数据/事件不足时停止新增信号并标注不完整，不伪造完整结果。",
                "- 提醒/计划不等于成交；只有录入成交后才改变现金与持仓。",
            ]
        )
        return "\n".join(lines)

    def _next_condition(self, decision: PositionDecision) -> str:
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

    def _html(
        self,
        day: str,
        result: StrategyPipelineResult | None,
        account: AccountState | None,
        decisions: list[PositionDecision],
        incomplete: list[str],
        data_notes: list[str],
        markdown: str,
    ) -> str:
        plan_rows = []
        if result is not None:
            for plan in result.plans:
                plan_rows.append(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                    "<td>{}</td><td>{}</td></tr>".format(
                        html.escape(plan.candidate.ts_code),
                        html.escape(
                            plan.strategy.value if plan.strategy else ""
                        ),
                        html.escape(plan.status.value),
                        html.escape(str(plan.entry_plan.entry_price)),
                        html.escape(str(plan.stop_plan.initial_stop)),
                        html.escape(str(plan.position_plan.shares)),
                    )
                )
        decision_rows = []
        for decision in decisions:
            decision_rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(decision.ts_code),
                    html.escape(
                        decision.primary_strategy.value
                        if decision.primary_strategy
                        else "待补全"
                    ),
                    html.escape(decision.action.value),
                    html.escape(decision.reason_code),
                    html.escape(str(decision.suggested_shares)),
                    html.escape(str(decision.current_stop or "待补全")),
                    html.escape(decision.explanation),
                )
            )
        incomplete_html = ""
        if incomplete:
            incomplete_html = (
                "<h2>数据不完整</h2><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in incomplete)
                + "</ul><p>未生成新增信号；已有持仓保护仍已执行。</p>"
            )
        if data_notes:
            incomplete_html += (
                "<h2>数据说明</h2><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in data_notes)
                + "</ul>"
            )
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>交易辅助日报 {day}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;
padding:0 1rem;line-height:1.6}}
table{{border-collapse:collapse;width:100%;margin-bottom:1rem}}
td,th{{border:1px solid #ddd;padding:.4rem;text-align:left}}
pre{{white-space:pre-wrap;background:#f7f7f7;padding:1rem}}
</style>
</head><body><h1>交易辅助日报 {day}</h1>
<p><strong>研究/决策辅助，不构成投资建议；所有成交由人工执行。</strong></p>
{incomplete_html}
<h2>持仓行动建议</h2>
<table><thead><tr><th>代码</th><th>策略</th><th>动作</th><th>原因码</th>
<th>建议股数</th><th>保护线</th><th>说明</th></tr></thead>
<tbody>{''.join(decision_rows)}</tbody></table>
<h2>计划</h2>
<table><thead><tr><th>代码</th><th>策略</th><th>状态</th><th>入场</th>
<th>止损</th><th>股数</th></tr></thead>
<tbody>{''.join(plan_rows)}</tbody></table>
<h2>完整文本</h2><pre>{html.escape(markdown)}</pre>
</body></html>"""
