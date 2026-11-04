"""Chinese labels and a small Markdown renderer for daily reports.

Only the presentation layer lives here.  Internal enums, API payloads, strategy
rules and risk thresholds remain unchanged.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

STRATEGY_NAMES = {
    "A": "趋势回踩",
    "B": "涨停承接",
    "C": "平台突破",
    "D": "超跌修复",
    "E": "区间交易",
    "F": "事件驱动",
}

SUB_MODE_NAMES = {
    "a_shallow": "浅回踩",
    "a_deep": "深回踩",
    "b1": "盘后承接",
    "b2": "盘中承接",
    "c1": "突破参与",
    "c2": "突破后回踩",
    "d_uptrend": "上升趋势超跌",
    "d_downtrend": "下降趋势超跌",
    "e_range": "区间交易",
    "f_event": "事件驱动",
    "f_earnings": "业绩事件",
    "f_buyback": "回购/增持事件",
    "f_contract": "合同/中标事件",
    "f_policy": "政策事件",
}

TREND_LABELS = {"attack": "进攻", "neutral": "中性", "defensive": "防守"}
TREND_NOTES = {
    "attack": "指数在 MA20 上方且市场宽度达到进攻阈值，趋势策略可按正常权限参与。",
    "neutral": "指数与市场宽度介于进攻和防守之间，趋势策略通常按中性权限运行。",
    "defensive": "指数低于 MA60 或市场宽度低于防守阈值，趋势策略权限上限为 0，暂停新计划。",
}

SENTIMENT_LABELS = {
    "freeze": "冰点",
    "start": "启动",
    "ferment": "发酵",
    "climax": "高潮",
    "ebb": "退潮",
}
SENTIMENT_NOTES = {
    "freeze": "涨停数量少、跌停/炸板偏多、红盘率偏低，情绪处于低位。",
    "start": "涨停数量、最高连板、炸板率和红盘率落入启动区间，情绪从低位回暖，尚未进入发酵或高潮。",
    "ferment": "涨停和连板增加、炸板率较低、红盘率较高，赚钱效应开始扩散。",
    "climax": "涨停/连板高度和红盘率处于高位，但炸板率上升或涨停溢价回落，需防退潮。",
    "ebb": "涨停减少、跌停/炸板增加、连板高度下降、红盘率偏低，情绪转弱。",
}

PERMISSION_LABELS = {
    "normal": "正常",
    "reduced": "降低风险额度",
    "paused": "暂停",
}
PERMISSION_NOTES = {
    "normal": "按正常风险预算运行，可生成新计划。",
    "reduced": "按降低后的风险预算运行，仓位/数量会被压缩。",
    "paused": "不分配新风险预算，不生成新计划。",
}

BREADTH_LABELS = {"broad": "偏宽", "normal": "中性", "narrow": "偏窄"}
BREADTH_NOTES = {
    "broad": "市场宽度达到进攻阈值，上涨股票占比高。",
    "normal": "市场宽度处于中性区间。",
    "narrow": "市场宽度低于防守阈值，上涨股票占比偏低。",
}

VOLATILITY_LABELS = {"calm": "偏低", "normal": "正常", "stress": "压力"}
VOLATILITY_NOTES = {
    "calm": "指数波动分位偏低。",
    "normal": "指数波动处于正常区间。",
    "stress": "指数波动分位达到压力阈值。",
}

LIQUIDITY_LABELS = {"normal": "正常", "dry": "偏枯竭"}
LIQUIDITY_NOTES = {
    "normal": "市场成交额处于正常水平。",
    "dry": "市场成交额相对近期明显萎缩。",
}

GLOBAL_RISK_LABELS = {
    "normal": "正常",
    "pause_new": "暂停新增",
    "reduce": "降低风险",
    "clear": "清仓风险复核",
}
GLOBAL_RISK_NOTES = {
    "normal": "未触发全局风险开关。",
    "pause_new": "全局风险开关暂停新增计划。",
    "reduce": "全局风险开关要求降低风险额度。",
    "clear": "指数出现大幅下跌，进入清仓风险复核。",
}

DEGRADED_LABELS = {
    "initial_stage": "情绪阶段使用初始估计（历史样本不足），阶段判断可能不稳定。",
    "st_namechange_missing": "ST/名称变更数据缺失，股票池过滤可能不完整。",
    "industry_stock_basic": "行业映射使用降级来源，行业归属可能不完整。",
    "limit_list_d": (
        "涨跌停明细数据为空或未同步，涨跌停家数、连板等统计可能不完整；"
        "这是数据降级，不代表当天没有涨跌停。"
    ),
    "index_fallback": "主指数数据不可用，已改用备用指数。",
    "stock_reference_incomplete": "股票基础信息不完整，存在参考范围外的交易代码。",
    "suspend_d_missing": "停牌数据缺失，股票池可能包含停牌股。",
}
DEGRADED_SHORT_LABELS = {
    "initial_stage": "情绪初始估计",
    "st_namechange_missing": "ST/名称变更数据缺失",
    "industry_stock_basic": "行业映射降级",
    "limit_list_d": "涨跌停明细缺失",
    "index_fallback": "使用备用指数",
    "stock_reference_incomplete": "股票基础信息不完整",
    "suspend_d_missing": "停牌数据缺失",
}

ACTION_LABELS = {
    "hold": "持有",
    "conditional_add": "条件加仓",
    "reduce": "减仓",
    "exit": "清仓",
    "wait_execution": "等待委托成交",
    "data_anomaly": "数据异常",
}
REASON_LABELS = {
    "hard_stop": "触及硬止损",
    "event_invalidated": "事件被撤回、证伪或转负",
    "event_expired": "事件窗口到期未兑现",
    "time_stop": "时间退出条件触发",
    "targets_complete": "全部止盈目标已完成",
    "target_1": "触及分批止盈目标 1",
    "target_2": "触及分批止盈目标 2",
    "target_3": "触及分批止盈目标 3",
    "hold": "未触发退出条件",
    "conditional_add": "满足条件加仓观察",
    "data_unavailable": "行情缺失或不可用",
    "stop_missing": "保护线缺失或无效",
    "invalid_price": "行情价格无效",
    "position_incomplete": "持仓记录不完整",
    "quote_stale": "行情已过期",
    "awaiting_execution": "已有卖出委托待成交",
    "new_high_history_missing": "缺少最近新高日期，无法判断时间退出",
    "position_recovered": "风险状态恢复",
    "near_stop": "接近保护线",
}
VALUATION_LABELS = {
    "ok": "正常",
    "stale": "已过期",
    "missing": "缺失",
    "unknown": "未知",
}
DECISION_SOURCE_LABELS = {
    "daily": "盘后日线",
    "intraday": "盘中实时",
    "persisted": "盘中快照",
}
STALE_REASON_LABELS = {
    "decision_time_missing": "决策缺少数据时间",
    "intraday_quote_expired": "盘中报价已过期",
    "intraday_quote_from_earlier_day": "盘中报价来自更早交易日",
    "intraday_snapshot_before_latest_close": "盘中快照早于最新收盘",
    "intraday_decision_missing": "盘中决策缺失，仅有上一交易日盘后结果",
    "post_close_data_missing": "盘后数据尚未处理",
    "daily_data_stale": "日线数据过期",
}
PLAN_STATUS_LABELS = {
    "DRAFT": "草稿",
    "OBSERVING": "观察中",
    "PENDING_TRIGGER": "等待触发",
    "CONFIRMED": "已确认",
    "PENDING_EXECUTION": "待执行",
    "PENDING_ENTRY": "待入场",
    "PARTIAL_FILL": "部分成交",
    "REJECTED": "已拒绝",
    "OPEN": "持仓中",
    "CLOSED": "已关闭",
    "EXPIRED": "已过期",
    "CANCELLED": "已取消",
    "INVALIDATED": "已失效",
}
TARGET_STATUS_LABELS = {
    "pending": "待触发",
    "partial": "部分完成",
    "done": "已完成",
}
ORDER_STATUS_LABELS = {
    "pending": "已报待成交",
    "partial": "部分成交",
    "cancel_requested": "撤单待确认",
    "filled": "已成交",
    "cancelled": "已撤销",
    "rejected": "已拒绝",
}
CANDIDATE_STAGE_LABELS = {"observing": "观察中", "confirmed": "已确认"}

PERMISSION_REASON_LABELS = {
    "market conditions not permitted": (
        "当前市场趋势、宽度、情绪和波动组合不满足该策略的正常或"
        "降低风险额度条件"
    ),
    "market conditions reduced": "当前市场条件只满足降低风险额度条件",
    "global risk switch": "全局风险开关已暂停新增",
    "global reduce-risk switch": "全局风险开关要求降低风险额度",
    "strategy disabled": "该策略在配置中已禁用",
    "minute/realtime data degraded": "分钟/实时数据降级",
    "strategy permission paused": "策略权限为暂停",
    "candidate permission paused": "候选权限为暂停",
    "risk factor is zero": "策略风险系数为 0",
}

GLOBAL_RISK_REASON_LABELS = {
    "data fault: no new signals": "数据异常，停止新增信号",
    "index one-day drop <= -8%: clear-risk review": "指数单日跌幅达到 8%，进入清仓风险复核",
    "index one-day drop <= -5%: pause new": "指数单日跌幅达到 5%，暂停新增",
    "volatility stress + narrow breadth: reduce risk": "波动压力叠加宽度偏窄，降低风险",
}

_STOP_DISTANCE_RE = re.compile(
    r"^(?P<strategy>[A-F]) stop distance "
    r"(?P<distance>\d+(?:\.\d+)?)% exceeds (?P<limit>\d+(?:\.\d+)?)%$"
)
_MAX_STRATEGY_POSITIONS_RE = re.compile(r"^max ([A-F]) positions reached$")
_DECLINE_DIFFUSION_RE = re.compile(
    r"^decline diffusion (?P<value>\d+(?:\.\d+)?): reduce risk$"
)

REJECTION_LABELS = {
    "entry price or ATR unavailable": "因缺少可用的拟入场价或 ATR，无法计算风险与仓位",
    "non-positive R": "因拟入场价与止损价之差不为正，风险 R 无效",
    "range upper target unavailable": "因区间上沿目标不可用，无法计算风险收益比",
    "nearest resistance is closer than 1.5R": "因最近可靠阻力位距入场价不足 1.5R，风险收益比不足",
    "risk configuration incomplete": "因真实账户尚未完成风险配置",
    "account data stale; reconcile positions before new plans": (
        "因账户数据已过期，需先人工核对持仓"
    ),
    "trading halted: manual restart required": "因交易已暂停，需人工解除",
    "strategy permission paused": "因策略权限为暂停",
    "candidate permission paused": "因候选权限为暂停",
    "risk factor is zero": "因风险系数为 0",
    "drawdown circuit breaker active": "因账户回撤达到停止开仓阈值",
    "valuation incomplete; reconcile prices before new plans": (
        "因估值不完整，需先补全行情并核对价格"
    ),
    "risk amount is zero": "因风险预算为 0",
    "entry price must exceed stop": "因拟入场价必须高于止损价",
    "duplicate pending plan": "因该股票已有待执行计划，避免重复占用资金",
    "maximum add-ons reached": "因已达到最大加仓次数",
    "add-on requires the existing position to be profitable": "因加仓要求现有持仓盈利",
    "add-on blocked by existing hard risk": "因现有持仓已触发硬风险，禁止加仓",
    "add-on blocked: existing position needs reconciliation": "因现有持仓待补全，禁止加仓",
    "add-on would lower the existing protection line": "因加仓计划的止损低于现有保护线",
    "max positions reached": "因已达到最大持仓数量",
    "max B positions reached": "因 B 策略持仓数量已达上限",
    "max industry positions reached": "因该行业持仓数量已达上限",
    "correlation above threshold": "因与现有持仓的相关性超过阈值",
    "cash reserve would be breached": "因会突破最低现金保留比例",
    "portfolio heat limit reached": "因组合风险预算已达上限",
    "position below one lot": "因计算股数不足 100 股，无法建立最小交易单位",
}

def _raw(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _label(mapping: dict[str, str], kind: str, value: Any) -> str:
    if value is None or value == "":
        return "未知"
    key = _raw(value)
    label = mapping.get(key)
    if label is not None:
        return label
    logger.warning("Unknown report %s value: %r", kind, value)
    return f"未识别（原始值：{key}）"


def _note(mapping: dict[str, str], kind: str, value: Any) -> str:
    if value is None or value == "":
        return "未知"
    key = _raw(value)
    note = mapping.get(key)
    if note is not None:
        return note
    logger.warning("Unknown report %s value: %r", kind, value)
    return f"未识别的{kind}（原始值：{key}），请查看日志"


def strategy_name(value: Any) -> str:
    return _label(STRATEGY_NAMES, "策略", value)


def strategy_label(value: Any) -> str:
    if value is None or value == "":
        return "策略待补全"
    key = _raw(value)
    name = STRATEGY_NAMES.get(key)
    if name is None:
        logger.warning("Unknown report strategy value: %r", value)
        return f"{key} 未识别策略"
    return f"{key} {name}"


def sub_mode_label(value: Any) -> str:
    return _label(SUB_MODE_NAMES, "策略子模式", value) if value else "未指定子模式"


def trend_label(value: Any) -> str:
    return _label(TREND_LABELS, "趋势", value)


def trend_note(value: Any) -> str:
    return _note(TREND_NOTES, "趋势", value)


def sentiment_label(value: Any) -> str:
    return _label(SENTIMENT_LABELS, "情绪阶段", value)


def sentiment_note(value: Any) -> str:
    return _note(SENTIMENT_NOTES, "情绪阶段", value)


def permission_label(value: Any) -> str:
    return _label(PERMISSION_LABELS, "策略权限", value)


def permission_note(value: Any) -> str:
    return _note(PERMISSION_NOTES, "策略权限", value)


def breadth_label(value: Any) -> str:
    return _label(BREADTH_LABELS, "市场宽度", value)


def breadth_note(value: Any) -> str:
    return _note(BREADTH_NOTES, "市场宽度", value)


def volatility_label(value: Any) -> str:
    return _label(VOLATILITY_LABELS, "波动状态", value)


def volatility_note(value: Any) -> str:
    return _note(VOLATILITY_NOTES, "波动状态", value)


def liquidity_label(value: Any) -> str:
    return _label(LIQUIDITY_LABELS, "流动性状态", value)


def liquidity_note(value: Any) -> str:
    return _note(LIQUIDITY_NOTES, "流动性状态", value)


def global_risk_label(value: Any) -> str:
    return _label(GLOBAL_RISK_LABELS, "全局风险开关", value)


def global_risk_note(value: Any) -> str:
    return _note(GLOBAL_RISK_NOTES, "全局风险开关", value)


def degraded_note(value: Any) -> str:
    if value is None or value == "":
        return "无"
    key = _raw(value)
    note = DEGRADED_LABELS.get(key)
    if note is not None:
        return note
    logger.warning("Unknown report degraded flag: %r", value)
    return f"未识别的数据降级项（原始值：{key}），请查看日志"


def degraded_short(value: Any) -> str:
    key = _raw(value)
    label = DEGRADED_SHORT_LABELS.get(key)
    if label is not None:
        return label
    logger.warning("Unknown report degraded flag: %r", value)
    return "未识别降级项"


def permission_reason(value: Any) -> str:
    key = _raw(value)
    label = PERMISSION_REASON_LABELS.get(key)
    if label is not None:
        return label
    logger.warning("Unknown report permission reason: %r", value)
    return f"未识别的权限原因（原始值：{key}），请查看日志"


def global_risk_reason(value: Any) -> str:
    key = _raw(value)
    label = GLOBAL_RISK_REASON_LABELS.get(key)
    if label is not None:
        return label
    match = _DECLINE_DIFFUSION_RE.match(key)
    if match:
        return f"下跌扩散指标为 {float(match.group('value')):.1f}，降低风险"
    logger.warning("Unknown report global-risk reason: %r", value)
    return f"未识别的全局风险原因（原始值：{key}），请查看日志"


def action_label(value: Any) -> str:
    return _label(ACTION_LABELS, "持仓动作", value)


def reason_label(value: Any) -> str:
    return _label(REASON_LABELS, "决策原因", value)


def valuation_label(value: Any) -> str:
    return _label(VALUATION_LABELS, "估值状态", value)


def decision_source_label(value: Any) -> str:
    return _label(DECISION_SOURCE_LABELS, "决策来源", value)


def stale_reason_label(value: Any) -> str:
    return _label(STALE_REASON_LABELS, "过期原因", value)


def plan_status_label(value: Any) -> str:
    return _label(PLAN_STATUS_LABELS, "计划状态", value)


def target_status_label(value: Any) -> str:
    return _label(TARGET_STATUS_LABELS, "目标状态", value)


def order_status_label(value: Any) -> str:
    return _label(ORDER_STATUS_LABELS, "委托状态", value)


def candidate_stage_label(value: Any) -> str:
    return _label(CANDIDATE_STAGE_LABELS, "候选阶段", value)


def rejection_clause(reason: str) -> str:
    text = str(reason).strip()
    match = _STOP_DISTANCE_RE.match(text)
    if match:
        distance = float(match.group("distance"))
        limit = float(match.group("limit"))
        return (
            f"的拟入场价到止损价的距离为 {distance:.2f}%，"
            f"超过 {limit:.2f}% 的上限"
        )
    label = REJECTION_LABELS.get(text)
    if label is not None:
        return label
    match = _MAX_STRATEGY_POSITIONS_RE.match(text)
    if match:
        return f"因 {match.group(1)} 策略持仓数量已达上限"
    if text.startswith("cannot schedule expiry:"):
        detail = text.split(":", 1)[1].strip()
        return f"因无法按交易日历安排计划有效期（{detail}）"
    if text.startswith("add-on not allowed for strategy "):
        return "因该策略不允许加仓"
    if text.startswith("entry price is invalid:"):
        return "因拟入场价无效"
    if text.startswith("entry price must be finite and positive"):
        return "因拟入场价不是有限正数"
    if text.startswith("R must be finite and positive"):
        return "因风险 R 不是有限正数"
    if text.startswith("initial stop"):
        return "因初始止损价格无效或高于入场价"
    if text.startswith("target arithmetic failed:"):
        return "因止盈目标计算失败"
    if text.startswith("nearest resistance "):
        return "因最近可靠阻力位距入场价不足 1.5R，风险收益比不足"
    if text.startswith("A targets are not ordered"):
        return "因 A 策略止盈目标顺序无效"
    if text.startswith("target ") and "must be above entry" in text:
        return "因止盈目标必须高于拟入场价"
    if text.startswith("target ") and "has invalid R multiple" in text:
        return "因止盈目标的 R 倍数无效"
    if " must be in (0, 1), got " in text:
        return "因目标比例必须在 0 和 1 之间"
    logger.warning("Unknown report rejection reason: %r", reason)
    return f"因未识别的拒绝原因（原始值：{text}），请查看日志"


def rejection_sentence(
    strategy_value: Any,
    confirmed_count: int,
    rejected: list[tuple[str, str]],
    plan_codes: Iterable[str] = (),
) -> str:
    """Build a complete Chinese sentence for rejected confirmed candidates."""
    if not rejected:
        return ""
    prefix = (
        f"{strategy_label(strategy_value)}策略有 {confirmed_count} 只股票"
        "通过策略确认，但 "
    )
    details = "；".join(
        f"{code} {rejection_clause(reason)}" for code, reason in rejected
    )
    if set(code for code, _ in rejected).isdisjoint(set(plan_codes)):
        return f"{prefix}{details}，未生成交易计划。"
    return f"{prefix}{details}，上述股票未生成交易计划。"


def format_datetime(value: Any) -> str:
    if value is None:
        return "未知"
    text = str(value)
    if text.endswith("+00:00"):
        text = text[:-6]
    if "." in text:
        head, tail = text.split(".", 1)
        if tail[:6].isdigit():
            timezone_part = ""
            for marker in ("+", "Z"):
                if marker in tail:
                    timezone_part = tail[tail.index(marker):]
                    break
            text = head + timezone_part
    return text


def inline_markdown(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells
    )


def _render_list(lines: list[str], start: int) -> tuple[str, int]:
    items: list[str] = []
    index = start
    while index < len(lines):
        match = re.match(r"^\s*[-*]\s+(?P<text>.+)$", lines[index])
        if not match:
            break
        items.append(match.group("text"))
        index += 1
    if not items:
        return "", start
    return (
        "<ul>" + "".join(f"<li>{inline_markdown(item)}</li>" for item in items) + "</ul>",
        index,
    )


def markdown_to_html(markdown: str) -> str:
    """Render the limited Markdown emitted by ReportService.

    Supports headings, blockquotes, ordered/unordered lists, pipe tables,
    bold/inline code and paragraphs.  The source is fully HTML-escaped.
    """
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped == "---":
            output.append("<hr>")
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            output.append(
                f"<h{level}>{inline_markdown(heading.group(2))}</h{level}>"
            )
            index += 1
            continue
        if stripped.startswith("> "):
            quoted: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("> "):
                quoted.append(lines[index].strip()[2:])
                index += 1
            output.append(
                "<blockquote><p>"
                + "<br>".join(inline_markdown(item) for item in quoted)
                + "</p></blockquote>"
            )
            continue
        if stripped.startswith("|"):
            header = _split_table_row(stripped)
            index += 1
            if index < len(lines) and _is_table_separator(lines[index]):
                index += 1
                rows: list[list[str]] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append(_split_table_row(lines[index]))
                    index += 1
                output.append('<div class="table-wrap"><table><thead><tr>')
                output.extend(
                    f"<th>{inline_markdown(cell)}</th>" for cell in header
                )
                output.append("</tr></thead><tbody>")
                for row in rows:
                    output.append("<tr>")
                    output.extend(
                        f"<td>{inline_markdown(cell)}</td>" for cell in row
                    )
                    output.append("</tr>")
                output.append("</tbody></table></div>")
                continue
            # A stray pipe line is still rendered as text, not swallowed.
            output.append(f"<p>{inline_markdown(stripped)}</p>")
            continue
        if re.match(r"^\s*[-*]\s+", line):
            list_html, index = _render_list(lines, index)
            output.append(list_html)
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items: list[str] = []
            while index < len(lines):
                match = re.match(r"^\s*\d+\.\s+(.+)$", lines[index])
                if not match:
                    break
                items.append(match.group(1))
                index += 1
            output.append(
                "<ol>"
                + "".join(f"<li>{inline_markdown(item)}</li>" for item in items)
                + "</ol>"
            )
            continue
        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index]
            current_stripped = current.strip()
            if not current_stripped:
                break
            if (
                current_stripped.startswith("#")
                or current_stripped.startswith("> ")
                or current_stripped.startswith("|")
                or current_stripped == "---"
                or re.match(r"^\s*[-*]\s+", current)
                or re.match(r"^\s*\d+\.\s+", current)
            ):
                break
            paragraph.append(current_stripped)
            index += 1
        if paragraph:
            output.append(
                "<p>" + "<br>".join(inline_markdown(item) for item in paragraph) + "</p>"
            )
    return "\n".join(output)
