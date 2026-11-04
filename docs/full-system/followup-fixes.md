# 第二轮专项修复记录（每日分析链路）

> 审查基线：`97afc8af316995aa1202bd77498f54a40420a79e`
> 日期：2026-11-02
> 原则：继续人工交易；没有券商下单接口；未部署生产、未推送/合并。

## 1. 浏览器、API、SSE 与后台决策统一

**根因**：页面、API、SSE 每次都用“最新处理后日线”重新计算并展示决策，
没有比较已持久化的盘中决策时间；盘后日线会覆盖更晚的盘中 9 元清仓决策，
重新显示上一交易日 10 元持有，并把行情标为正常。

**修复位置**：
- `app/portfolio/decisions.py`：新增 `current_all()`、`_choose_current()`、
  `_assess_current()`。按 `data_time` 选择最新决策；盘中决策过期或缺失时
  明确标记 `stale_reason`，持有类动作转为 `data_anomaly/quote_stale`，
  风险动作保留。
- `PositionDecision` 新增 `decision_source`、`market_phase`、`is_current`、
  `stale_reason`、`pending_stop`、`pending_stop_effective_at`、
  `last_new_high_on`、`days_since_new_high`、`covered_shares`。
- `app/api/app.py` 的页面上下文、`/api/positions`、`/api/positions/decisions`、
  `/api/events/stream` 全部改用 `current_all()`；SSE 权益/持仓市值改用
  账户统一估值，不再用决策价格重新拼权益。
- `positions.html` / `base.html` 展示决策来源、有效期、待生效保护线、
  委托覆盖数量与过期原因。

**回归测试**：
- `test_followup_review.py::test_current_decisions_prefer_newer_intraday_exit_over_daily_hold`
- `test_followup_review.py::test_expired_intraday_hold_becomes_data_anomaly_not_safe_hold`
- `test_api.py::test_api_sse_and_page_prefer_newer_intraday_decision`

## 2. 盘中账户估值同步

**根因**：`IntradayMonitor.check()` 只把新价格写回持仓，没有调用
`AccountService.revalue()`，权益、风险字段和估值状态停留在旧值。

**修复位置**：`app/jobs/intraday.py` 在持仓决策落库后、B2 审核前调用统一
`revalue()`；缺行情保留最后价格并标记 `valuation_incomplete`，不伪装实时。
`AccountService.revalue()` 是唯一估值入口。

**回归测试**：
- `test_followup_review.py::test_intraday_revalues_before_b2_and_keeps_missing_quote_valuation`
- 既有 `test_review_fixes.py::test_valuation_updates_equity_and_excludes_deposits_from_peak`

## 3. 保护线生效时点与幂等

**根因**：日线评估算出的“次日保护线”被立即写入 `current_stop`；同一天
重复评估时，新保护线反过来触发当天硬止损，导致持有变清仓。

**修复位置**：
- `positions` 新增 `pending_stop`、`pending_stop_data_time`、
  `pending_stop_effective_at`、`last_new_high_on`（迁移
  `c8e1f2a3b4c5_exit_timing_and_order_fills.py`）。
- `decisions._effective_stop()` 只在数据时间到达生效时间（下一交易日
  09:30）后把待生效保护线提升为当前保护线；同日重复评估不会提升。
- 日线评估只写待生效保护线，不覆盖当前保护线；盘中 tick 不生成新的
  待生效保护线，避免用脉冲价格改写盘后规则。
- `app/backtest/simulator.py` 使用同样的“收盘计算、下一交易日生效”顺序。

**回归测试**：
- `test_followup_review.py::test_daily_stop_is_effective_next_session_and_repeated_runs_are_idempotent`
- `test_followup_review.py::test_intraday_tick_does_not_create_a_next_session_stop`
- `test_exits.py::test_exit_review_tightens_protection_and_flags_target`
- `test_review_fixes.py::test_b_position_tightens_protection_and_suggests_target_reduction`
- `test_review_fixes.py::test_live_and_backtest_use_same_exit_rule_without_future_stop`

## 4. 交易日历覆盖与自动刷新

**根因**：`TradingCalendar.is_trading_day()` 把“日历未覆盖的日期”和
“明确休市”都返回 False；数据更新和调度器据此永久跳过周一。

**修复位置**：
- `app/core/calendar.py`：保留覆盖起止和收盘日，新增 `covers()`、
  `is_explicit_holiday()`；只有覆盖范围内且非开市日才是明确休市。
- `app/jobs/data_update.py`：覆盖不足时先刷新日历；刷新失败标记
  `unknown` 并继续，不返回 `skipped`；刷新后确认休市才跳过。参考数据
  刷新窗口按最长计划有效期扩展（至少 45 天），`_reference_ready()` 还会
  检查未来覆盖是否足够，不足则重新拉取。
- `app/jobs/scheduler.py`、`app/jobs/intraday.py` 使用同样的覆盖判断。

**回归测试**：
- `test_followup_review.py::test_calendar_distinguishes_coverage_from_confirmed_holiday`
- `test_followup_review.py::test_data_update_tries_calendar_refresh_when_coverage_ends_before_monday`
- `test_followup_review.py::test_data_update_marks_unknown_when_refresh_fails_instead_of_skipping`

## 5. B/D 时间退出

**根因**：`no_new_high` 实际比较“持仓以来最高价是否超过成本”，没有记录
最近一次有效新高日期，也没有按交易日计算间隔。

**修复位置**：
- `positions.last_new_high_on` 记录新高日期；只在日线收盘评估中更新，
  盘中 tick 不改写。
- `ExitObservation.days_since_new_high` 按交易日历计算；`evaluate_exit()`
  在 `days_since_new_high >= time_stop_days` 时时间退出；真正创新高会重置
  为 0。
- 回测 `SimPosition.last_new_high_on` 与交易日索引同步更新。

**回归测试**：
- `test_followup_review.py::test_no_new_high_uses_trading_days_since_last_high_and_resets_on_new_high`
- `test_followup_review.py::test_service_records_last_new_high_date_and_time_exit_resets`

## 6. 报价新鲜度统一

**根因**：同日但超过年龄阈值的报价返回 `status=stale` 却 `usable=True`，
导致健康汇总 `degraded=False`，与决策使用的 `PricePoint.usable` 矛盾。

**修复位置**：`app/jobs/intraday.py::_validate()` 过期报价统一返回
`usable=False`；`app/api/app.py::_quote_freshness()` 统一页面/API/SSE
口径；`DataHealthService` 增加 `quote_freshness` 并检查持仓估值状态。

**回归测试**：
- `test_followup_review.py::test_same_day_expired_quote_is_not_usable_and_health_matches`
- 既有 `test_review_fixes.py::test_intraday_rejects_stale_and_missing_time_and_keeps_other_positions`
- 既有 `test_review_fixes.py::test_intraday_missing_and_future_timestamps_are_unknown_and_holiday_skips`

## 7. 部分卖单不掩盖剩余硬止损

**根因**：任何卖出委托都会把动作改为 `wait_execution` 并把建议数量清零；
委托剩余量又直接使用原始 `shares`，部分成交时高估覆盖。

**修复位置**：
- `orders.filled_shares` 记录累计成交；`_execution_totals()` 用
  `shares - filled_shares` 计算真实剩余。
- `decisions.evaluate_position()` 保留底层 `exit/reduce` 风险动作，
  新增 `covered_shares` 和 `pending_sell_orders`（含价格、原始/已成交/
  剩余股数、状态），`suggested_shares` 只给未覆盖数量，并提示撤改单
  或补充卖出。

**回归测试**：
- `test_followup_review.py::test_small_pending_sell_order_does_not_hide_remaining_hard_stop_risk`
- `test_followup_review.py::test_fully_covered_exit_keeps_risk_action_with_zero_uncovered_quantity`

## 8. 计划取消与撤单确认分离

**根因**：取消计划直接改 `CANCELLED`，风险占用按计划状态释放；没有
“撤单请求未确认”状态，部分成交也没有剩余数量事实。

**修复位置**：
- `OrderStatus.CANCEL_REQUESTED`；`cancel_order()` 默认只记录撤单请求，
  `confirm_order_cancel()` 才释放；`reject_order_cancel()` 恢复实际状态；
  `record_order_fill()` 按累计成交更新剩余数量。
- `cancel_plan()` 只取消本地计划，不假设券商撤单成功；审计返回活动委托
  和保留数量。
- `PlanRepository.pending_exposures()` 改为“订单事实优先”：活动订单即使
  所属计划取消/过期，仍按剩余数量占用现金和风险；无订单时才按计划剩余
  数量占用。
- API 新增委托撤单/确认/拒绝/成交更新接口；CSV 委托模板新增
  `filled_shares`。
- 迁移新增 `orders.filled_shares`。

**回归测试**：
- `test_followup_review.py::test_plan_cancel_and_order_cancel_keep_exposure_until_confirmed`
- `test_followup_review.py::test_expired_plan_does_not_release_active_order_exposure`
- `test_portfolio.py::test_cancel_order_rejects_filled_and_is_idempotent`

## 9. 回测退出输入补齐

**根因**：`SimPosition.event_expiry`/`event_invalid` 未填充；
`ExitObservation.data_time` 为空导致事件到期永不触发；MA5、区间中点、
动态 ATR 从未计算，F/B/D/E 的共享退出规则缺少真实输入。

**修复位置**：
- `app/backtest/simulator.py`：
  - 入场时从候选证据/信号时间计算 `event_expiry`，并从历史事件更新
    `event_invalid`（撤回/负面且已核验，支持同一 `event_id` 的后期撤回）；
    若事件列表缺少该 `event_id` 或事件覆盖缺失，写入 `missing_data`，
    不默认事件持续有效。
  - 每个交易日使用滚动日线计算 MA5、ATR14；区间中点来自 E 候选证据。
  - `ExitObservation.data_time` 使用当日 15:00，事件到期按交易日历触发。
  - 待生效保护线下一个处理交易日提升，和实盘一致。
- `app/backtest/engine.py` 将交易日历和 `events_by_code` 注入模拟器。

**回归测试**：
- `test_followup_review.py::test_backtest_f_event_expiry_uses_known_event_date_and_data_time`
- `test_followup_review.py::test_backtest_f_retraction_invalidates_and_missing_events_is_labeled`
- `test_followup_review.py::test_backtest_ma5_range_and_atr_inputs_are_updated`
- 既有 `test_backtest.py` 与 `test_review_fixes.py` 回测用例

## 10. 缺行情/缺止损异常优先级

**根因**：数据检查在止损检查之前，缺行情且缺止损时返回
`data_unavailable`，掩盖了持仓本身无法保护的问题；测试依赖本地是否存在
日线，导致结果不稳定。

**修复位置**：`evaluate_exit()` 先检查初始/当前保护线，再检查行情可用性，
最后才做价格有效性、硬止损和事件/时间退出。缺行情但保护线存在仍返回
`data_unavailable`；两者都缺返回 `stop_missing`，与行情文件是否存在无关。

**回归测试**：
- `test_followup_review.py::test_stop_missing_has_priority_but_missing_quote_alone_is_data_unavailable`
- `tests/test_api.py::test_manual_execution_updates_positions`（不再依赖本地日线）

## 11. 每日分析专项验收

见 `docs/full-system/test-report.md` 的“第二轮验收”章节。实际运行
`daily --day 2026-11-02`，真实 Tushare 增量同步 5 个交易日，质量门通过，
数据更新状态 `ready`，报告写入 `data/reports/2026-11-02/daily.md`。

## 当前限制

- 仍未接券商下单/撤单接口；“委托”事实依赖 CSV 导入或后续券商对账。
- B2 分钟级真实盘中链路未在本次盘后验收中执行；相关测试使用注入数据。
- 浏览器端通过 TestClient/SSE 契约验证，未做真实浏览器自动化。
- 策略仍为研究状态；软件修复不代表盈利。
