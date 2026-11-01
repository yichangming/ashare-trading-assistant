"""Daily HTML/Markdown report generation."""

from __future__ import annotations

import html
from pathlib import Path

from app.core.full_config import BACKEND_ROOT
from app.pipeline_v2 import StrategyPipelineResult


class ReportService:
    def __init__(self, reports_root: Path | None = None):
        self.reports_root = reports_root or BACKEND_ROOT / "data" / "reports"

    def generate(self, result: StrategyPipelineResult) -> dict[str, str]:
        day = result.as_of.isoformat()
        directory = self.reports_root / day
        directory.mkdir(parents=True, exist_ok=True)
        markdown = self._markdown(result)
        html_text = self._html(result, markdown)
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

    def _markdown(self, result: StrategyPipelineResult) -> str:
        regime = result.regime
        lines = [
            f"# A 股交易辅助日报 — {result.as_of.isoformat()}",
            "",
            "> 研究/决策辅助，不构成投资建议；所有成交由人工执行。",
            "",
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
                    f"- {plan.candidate.ts_code} {plan.strategy.value if plan.strategy else ''}"
                    f" {plan.status.value} 入场={plan.entry_plan.entry_price}"
                    f" 止损={plan.stop_plan.initial_stop}"
                    f" 股数={plan.position_plan.shares}"
                )
        lines.extend(
            [
                "",
                "## 账户",
                "",
                f"- 权益：{result.account.equity}",
                f"- 现金：{result.account.cash}",
                f"- 持仓数：{len(result.account.positions)}",
                f"- 待成交计划：{len(result.account.pending_plans)}",
                "",
                "## 限制",
                "",
                "- B2 盘中模式需要 5 分钟数据；日报只反映日线盘后结果。",
                "- 数据/事件不足时，相关策略权限会降档或暂停。",
            ]
        )
        return "\n".join(lines)

    def _html(self, result: StrategyPipelineResult, markdown: str) -> str:
        # A compact, dependency-free rendering.  The Markdown file remains the
        # canonical machine-readable report; HTML is for browser sharing.
        rows = []
        for plan in result.plans:
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                "<td>{}</td><td>{}</td></tr>".format(
                    html.escape(plan.candidate.ts_code),
                    html.escape(plan.strategy.value if plan.strategy else ""),
                    html.escape(plan.status.value),
                    html.escape(str(plan.entry_plan.entry_price)),
                    html.escape(str(plan.stop_plan.initial_stop)),
                    html.escape(str(plan.position_plan.shares)),
                )
            )
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>交易辅助日报 {result.as_of}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;
padding:0 1rem;line-height:1.6}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:.4rem;text-align:left}}
pre{{white-space:pre-wrap;background:#f7f7f7;padding:1rem}}
</style>
</head><body><h1>交易辅助日报 {result.as_of}</h1>
<p><strong>研究/决策辅助，不构成投资建议；所有成交由人工执行。</strong></p>
<table><thead><tr><th>代码</th><th>策略</th><th>状态</th><th>入场</th><th>止损</th><th>股数</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>完整文本</h2><pre>{html.escape(markdown)}</pre>
</body></html>"""
