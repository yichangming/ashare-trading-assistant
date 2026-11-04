from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.domain.enums import (
    GlobalRiskMode,
    PermissionMode,
    PositionAction,
    SentimentStage,
    Strategy,
    SubMode,
    Template,
    TrendRegime,
    ValuationStatus,
)
from app.domain.models import (
    AccountState,
    Candidate,
    MarketRegime,
    PositionDecision,
    RegimeIndicators,
    StrategyPermission,
    TemplatePolicy,
)
from app.jobs.report_text import markdown_to_html
from app.jobs.reports import ReportService
from app.pipeline_v2 import RejectedCandidate

SHANGHAI = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 11, 3)


def _candidate(
    code: str,
    strategy: Strategy,
    sub_mode: SubMode,
    *,
    stage: str = "observing",
    score: float = 50.0,
) -> Candidate:
    return Candidate(
        ts_code=code,
        trade_date=DAY,
        template=Template.B if strategy == Strategy.B else Template.A,
        setup_type="test",
        horizon="mid",
        sector="电子",
        leader_rank=1,
        leader_score=70.0,
        score=score,
        reasons=["测试"],
        features={"close": 10.0},
        sentiment_stage=SentimentStage.START,
        strategy_version="sv",
        data_version="dv",
        strategy=strategy,
        sub_mode=sub_mode,
        evidence={"stage": stage},
        data_time=datetime(2026, 11, 3, 15, 0, tzinfo=SHANGHAI),
    )


def _regime(*, degraded: list[str] | None = None) -> MarketRegime:
    permissions = {
        Strategy.A: StrategyPermission(
            strategy=Strategy.A,
            mode=PermissionMode.PAUSED,
            risk_factor=0.0,
            reasons=["market conditions not permitted"],
        ),
        Strategy.B: StrategyPermission(
            strategy=Strategy.B,
            mode=PermissionMode.PAUSED,
            risk_factor=0.0,
            reasons=["market conditions not permitted"],
        ),
        Strategy.C: StrategyPermission(
            strategy=Strategy.C,
            mode=PermissionMode.PAUSED,
            risk_factor=0.0,
            reasons=["market conditions not permitted"],
        ),
        Strategy.D: StrategyPermission(
            strategy=Strategy.D,
            mode=PermissionMode.REDUCED,
            risk_factor=0.5,
            reasons=["market conditions reduced"],
        ),
        Strategy.E: StrategyPermission(
            strategy=Strategy.E,
            mode=PermissionMode.NORMAL,
            risk_factor=1.0,
            reasons=[],
        ),
        Strategy.F: StrategyPermission(
            strategy=Strategy.F,
            mode=PermissionMode.REDUCED,
            risk_factor=0.5,
            reasons=["market conditions reduced"],
        ),
    }
    return MarketRegime(
        trade_date=DAY,
        trend_regime=TrendRegime.DEFENSIVE,
        sentiment_stage=SentimentStage.START,
        risk_factor=0.0,
        template_availability={
            Template.A: TemplatePolicy(
                enabled=False, mode="observe", max_positions=0
            ),
            Template.B: TemplatePolicy(
                enabled=False, mode="observe", max_positions=0
            ),
        },
        indicators=RegimeIndicators(
            limit_up_count=132,
            limit_down_count=11,
            max_board_height=5,
            promotion_rate=0.18,
            blow_rate=0.20,
            red_rate=0.70,
            limit_up_premium=-0.02,
            breadth=0.198,
            index_close=5034.5,
            index_ma20=5417.0,
            index_ma60=5720.5,
        ),
        degraded_flags=degraded or ["limit_list_d"],
        data_version="dv",
        strategy_permissions=permissions,
        global_risk=GlobalRiskMode.NORMAL,
        breadth_label="narrow",
        volatility_label="normal",
        liquidity_label="normal",
    )


def _result(*, degraded: list[str] | None = None):
    confirmed = _candidate(
        "301563.SZ", Strategy.D, SubMode.D_UPTREND, stage="confirmed", score=47.6
    )
    candidates = [
        _candidate("002833.SZ", Strategy.D, SubMode.D_UPTREND, score=55.3),
        confirmed,
        _candidate("600845.SH", Strategy.E, SubMode.E_RANGE, score=70.9),
    ]
    return SimpleNamespace(
        as_of=DAY,
        regime=_regime(degraded=degraded),
        candidates=candidates,
        observations=[item for item in candidates if item is not confirmed],
        plans=[],
        rejected=[
            RejectedCandidate(
                candidate=confirmed,
                reason="D stop distance 10.64% exceeds 8.00%",
            )
        ],
        account=AccountState(
            equity=Decimal("1000000"),
            cash=Decimal("1000000"),
            peak_equity=Decimal("1000000"),
        ),
        by_strategy={"A": 0, "B": 0, "C": 0, "D": 2, "E": 1, "F": 0},
    )


def test_report_is_chinese_and_rejection_is_a_complete_sentence(tmp_path):
    report = ReportService(tmp_path).generate(
        _result(),
        as_of=DAY,
        account=AccountState(
            equity=Decimal("1000000"),
            cash=Decimal("1000000"),
            peak_equity=Decimal("1000000"),
        ),
        decisions=[],
        position_reviewed=True,
    )
    markdown = report["markdown"]
    html = report["html"]
    assert "## 今日摘要" in markdown
    assert "防守" in markdown
    assert "情绪阶段：启动" in markdown
    assert "A 趋势回踩" in markdown
    assert "D 超跌修复" in markdown
    assert "暂停" in markdown
    assert "降低风险额度" in markdown
    assert "风险预算乘数，不是持仓比例" in markdown
    assert "涨跌停明细缺失" in markdown
    assert "当前账户无持仓，无需操作提醒" in markdown
    assert (
        "D 超跌修复策略有 1 只股票通过策略确认，但 "
        "301563.SZ 的拟入场价到止损价的距离为 10.64%，"
        "超过 8.00% 的上限，未生成交易计划。"
    ) in markdown
    for english in (
        "defensive",
        "paused",
        "reduced",
        "market conditions",
        "stop distance",
        "stage=observing",
        "target_1",
        "limit_list_d",
    ):
        assert english not in markdown
    assert "<h1>" in html
    assert "<h2>今日摘要</h2>" in html
    assert "<table>" in html
    assert "<ul>" in html
    assert "<pre>" not in html
    assert "## 今日摘要" not in html
    assert "> 研究/决策辅助" not in html


def test_report_does_not_claim_no_alert_without_position_review(tmp_path):
    service = ReportService(tmp_path)
    no_review = service.generate(
        _result(),
        as_of=DAY,
        decisions=[],
        position_reviewed=False,
    )["markdown"]
    assert "持仓监控未运行或缺少持仓结果" in no_review
    assert "无需操作提醒" not in no_review

    hold = PositionDecision(
        ts_code="000001.SZ",
        sector="银行",
        primary_strategy=Strategy.A,
        sub_mode=SubMode.A_SHALLOW,
        action=PositionAction.HOLD,
        reason_code="hold",
        explanation="未触发退出条件。",
        total_shares=1000,
        available_shares=1000,
        current_stop=Decimal("9.5"),
        valuation_status=ValuationStatus.OK,
        data_time=datetime(2026, 11, 3, 15, 0, tzinfo=SHANGHAI),
        decision_source="daily",
    )
    reviewed = service.generate(
        _result(),
        as_of=DAY,
        decisions=[hold],
        position_reviewed=True,
    )["markdown"]
    assert "1 只持仓均无操作提醒" in reviewed
    assert "持仓监控未运行" not in reviewed

    anomaly = hold.model_copy(
        update={
            "action": PositionAction.DATA_ANOMALY,
            "reason_code": "data_unavailable",
            "explanation": "行情缺失，不能判定为安全持有。",
        }
    )
    risk = service.generate(
        _result(),
        as_of=DAY,
        decisions=[anomaly],
        position_reviewed=True,
    )["markdown"]
    assert "1 只持仓需要处理" in risk
    assert "数据异常 1" in risk
    assert "均无操作提醒" not in risk


def test_unknown_degraded_flag_is_chinese_and_logged(tmp_path):
    from unittest.mock import patch

    with patch("app.jobs.report_text.logger.warning") as warning:
        report = ReportService(tmp_path).generate(
            _result(degraded=["future_unknown_flag"]),
            as_of=DAY,
            decisions=[],
            position_reviewed=True,
        )
    assert "未识别的数据降级项（原始值：future_unknown_flag）" in report["markdown"]
    assert any(
        call.args
        and call.args[0] == "Unknown report degraded flag: %r"
        and call.args[1] == "future_unknown_flag"
        for call in warning.call_args_list
    )


def test_markdown_renderer_outputs_real_html():
    rendered = markdown_to_html(
        "# 标题\n\n> 引用\n\n- 项目\n\n| 策略 | 权限 |\n|---|---|\n"
        "| A 趋势回踩 | 暂停 |\n"
    )
    assert "<h1>标题</h1>" in rendered
    assert "<blockquote>" in rendered
    assert "<ul><li>项目</li></ul>" in rendered
    assert '<div class="table-wrap"><table>' in rendered
    assert "| 策略 | 权限 |" not in rendered
    assert "# 标题" not in rendered
