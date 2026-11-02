# 上一轮实现问题修复记录

> 审查基线：`0edeb3dcda2d28a1e5e9d2852f95b688141acd59`
> 修复分支：`fix/full-system-review`
> 日期：2026-11-02
> 原则：所有退出/减仓建议仍由人工执行；没有券商下单接口。软件验收与策略盈利分开。

## 1. 统一持仓行动决策

**根因**：持仓页只按“距止损”显示持有/接近止损；日线退出、盘中监控、
回测和报告各有一套逻辑，规则文本还被当作执行依据。

**修复**：
- 新增 `app/portfolio/exit_rules.py`：结构化 `ExitRules` 和纯函数
  `evaluate_exit()`。硬止损、事件失效、时间退出、分批止盈、移动保护线
  按固定优先级执行；新保护线只用收盘后数据计算，不回用于当日更早的最低价。
- 新增 `app/portfolio/decisions.py`：`PositionDecisionService` 输出
  `PositionDecision`，包含股票、主策略/子模式/规则版本、动作、原因码、
  说明、触发/数据时间、总持仓/可卖/建议/受限数量、保护线、目标、
  失效条件、确认状态、已成交/待成交数量和数据异常标记。
- `app/jobs/exits.py`、`app/jobs/intraday.py`、`app/api/app.py`、
  `app/jobs/reports.py` 和 `app/backtest/simulator.py` 均调用同一套规则。
- 数据缺失/过期输出 `data_anomaly`，不默认“安全持有”；硬风险清仓在
  条件加仓之前判断。

**证据**：
- `tests/test_review_fixes.py::test_b_position_tightens_protection_and_suggests_target_reduction`
- `test_missing_plan_or_stop_is_data_anomaly_not_safe_hold`
- `test_live_and_backtest_use_same_exit_rule_without_future_stop`
- `test_report_contains_position_actions_from_shared_decisions`
- `tests/test_api.py` 的持仓决策/SSE 字段断言。

## 2. 回测与实际退出规则不一致

**根因**：保护线上移只对 A/C 做了部分实现；B/D/E/F 的时间退出被简化为
“达到天数且低于成本”；计划文本描述与实际动作不一致。

**修复**：
- `StrategyPlanner` 生成 `ExitRules`（trailing/time_exit/time_stop_days/
  event_expiry_days），`TradePlan.exit_rules` 持久化；运行时不解析英文
  说明文字。
- 按策略实现：A/C/F `+1R 保本、+2R 上移 +1R、close-1.5ATR`；B/D
  `+1R 保本、+2R 后 MA5/1.5ATR`；E 区间中枢/ATR；F 事件到期/否定。
- 回测 `_check_exit` 直接调用 `evaluate_exit`；分批止盈按目标状态执行，
  同日止损优先；目标部分成交后剩余仓位继续保留。
- 复现 B 场景：成本 10、止损 9.5、最高 11.6；决策建议减仓且保护线
  上移，不再出现“计划说上移、检查无动作”。

**证据**：
- `test_b_position_tightens_protection_and_suggests_target_reduction`
- `test_live_and_backtest_use_same_exit_rule_without_future_stop`
- `test_full_chain_plan_partial_fills_valuation_reduce_and_close`
- `tests/test_exits.py` 的止损/上移/止盈测试。

## 3. 成交归属、初始 R 与加仓计数

**根因**：成交只写入 `plan_id`，没有继承策略/子模式/行业/规则版本；
`initial_r` 一直为 `None`；`add_index=len(position.lots)` 把首次买入的
多笔成交误算成多次加仓；无计划买入把买入价当作止损。

**修复**：
- 关联计划时校验账户与证券，继承 strategy/sub_mode/sector/version；
  价格偏离、计划过期、计划状态异常持久化为 `plan_deviation` 并记审计。
  默认允许记录真实成交（审计式），`strict_plan=True` 才拒绝。
- 初始批次的多笔成交共享 `add_index=0`、同一 `initial_r`；加仓按
  “不同计划/显式 add_on”只计一次，`PositionRow.add_on_count` 与
  `initial_shares` 持久化，不随 lot 卖出或删除重置；加仓不覆盖主策略和
  主计划。
- `initial_r = 实际成交价 - 计划初始止损`，写入 lot 和执行记录；旧数据
  缺少可靠字段时 `needs_reconciliation=True`，决策显示“待补全”，不把
  买入价默认成止损。
- 新增 Alembic 迁移 `b7c1d2e3f4a5_review_hardening.py`，新增归属、批次、
  估值、目标状态等列，并把旧持仓标记为待核对。

**证据**：
- `test_partial_initial_fills_are_one_batch_and_add_ons_are_counted`
- `test_plan_linkage_validates_owner_and_security_and_records_deviation`
- `tests/test_portfolio.py::test_manual_execution_respects_plan_price_and_expiry`
- `tests/test_migrations.py::test_review_migration_adds_columns_and_marks_old_positions`

## 4. 账户估值与每日运行顺序

**根因**：持仓价格更新后没有重算权益；日线任务在估值后把旧的
`result.account` 写回；`last_synced_at` 同时承担估值与对账含义；入出金
被当成收益；用户 `max_drawdown` 未参与计算。

**修复**：
- `AccountService.revalue()` 统一执行 `权益 = 现金 + Σ(现价×股数)`；
  缺行情保留最后价格并标记 `valuation_incomplete`，不清零。
- `last_valuation_at` 与 `last_reconciled_at` 分离；盘后任务不会宣称
  账户已对账。`mark_stale()` 只按人工核对时间判断真实账户。
- 入出金计入 `net_deposits`，峰值/回撤使用扣除净入金后的表现权益。
- 日线顺序改为：公司行动 → 可卖数量 → 核对时效 → 估值/持仓决策 →
  新增计划；不再调用 `save_state(result.account)`。
- 用户配置的 `max_drawdown` 参与 sizer/governor 与回撤 latch；估值不完整
  时真实账户拒绝新增，研究/模拟降半仓并标注。

**证据**：
- `test_valuation_updates_equity_and_excludes_deposits_from_peak`
- `test_daily_revalues_before_pipeline_and_does_not_overwrite_with_stale_state`
- `test_user_max_drawdown_participates_in_plan_governance`

## 5. B2 观察池与盘中计划生命周期

**根因**：`_watchlist` 只接受快照日期等于监控日期；正常上一交易日盘后
快照被拒绝；有效期按自然分钟计算；多信号共用旧账户快照；分钟缓存各自
计时。

**修复**：
- 按交易日历找上一交易日，读取该日盘后快照；周末/节假日自然跨越；
  不接受当日未完成日线。观察池日期、监控日期、信号时间、bar 完成时间
  分别记录。
- `stk_mins` 的 `trade_time` 按 bar 完成时间处理，不再额外减一根。
- `add_trading_minutes()` 跨午休计算 10 个交易分钟，最晚 15:00；
  `PlanRepository.expire_intraday()` 按精确 `expires_at` 失效。
- 同轮多个信号逐个使用更新后的账户（pending exposures/heat/cash），
  避免同一份旧余额。
- `TushareMinuteSource` 使用实例级锁和全局限频，缓存不再绕过总限频。
- 重启后观察池/信号去重/计划保持；失败/缺口计入 `data_issues`。

**证据**：
- `test_b2_previous_trading_day_watchlist_lunch_validity_and_restart`
- `test_b2_multiple_signals_use_updated_account_risk_budget`
- `test_minute_source_rate_limit_is_global_across_symbols`
- `tests/test_b2_intraday.py` 原有 B2 状态机测试。

## 6. 实时行情新鲜度与监控中断

**根因**：报价只取价格，日期/时区/新鲜度未校验；第一只失败即 `break`；
告警键包含价格，价格变化会刷屏。

**修复**：
- `Quote` 携带 `quote_time/fetched_at/source/delayed`；校验日期、时区、
  未来偏差、年龄阈值；缺失时间标记 `unknown`，不用当前时间伪装。
  旧日期/过期报价不更新价格、不触发止损。
- 单只失败继续检查其他持仓，返回逐股健康状态；分钟数据回退明确标记
  `minute_delayed`，无数据时 `missing`。
- 按持仓 `last_alert_status` 去重；恢复时生成 `position_recovered`。
- 交易日历过滤休市日。

**证据**：
- `test_intraday_rejects_stale_and_missing_time_and_keeps_other_positions`
- `test_intraday_deduplicates_alert_and_emits_recovery`
- `test_intraday_missing_and_future_timestamps_are_unknown_and_holiday_skips`
- `tests/test_jobs.py` 原有盘中测试。

## 7. 回测未成交计划被提前删除

**根因**：`process_day` 只在 `filled=True` 时保留订单，未成交且未过期的
计划当天即丢失。

**修复**：
- 订单状态区分 `pending/partial/filled/expired/invalidated/cancelled/
  cancel_unconfirmed`；未触发/部分成交保留到 `expires_on`。
- 剩余订单继续计入 `reserved_cash`、账户 pending exposure 和 heat。
- 新增强制复现：1/2 计划、1/3 未达买点、1/4 达到买点仍可成交。
- 入口/出口费用都计入交易净 PnL，并与权益曲线对账；`vol` 按 Tushare
  的“手”转换为 100 股；最小 100 股。
- 历史公司行动/退市的不足写入 `notes/missing_data`，不输出伪完整绩效。

**证据**：
- `test_backtest_keeps_unfilled_plan_until_expiry`
- `test_backtest_fees_reconcile_with_equity_and_cancel_is_explicit`
- `test_backtest_distinguishes_invalidation_and_unconfirmed_cancel`
- `tests/test_backtest.py` 的 T+1/涨跌停/部分成交/指标测试。

## 8. 自动数据更新贯通

**根因**：Scheduler 只调用 `daily`，而 `daily` 主要读本地数据；事件同步
未进入定时流程；无凭据时没有明确状态。

**修复**：
- 新增 `app/jobs/data_update.py`：交易日检查 → 参考/日历 → 增量日线 →
  清洗组装 → 质量门 → 结构化事件同步。阶段状态显式记录，
  已就绪阶段复用，失败/无凭据标记 `failed`/`unavailable`。
- Scheduler、Web 手动任务和 CLI `daily` 均启用数据更新；关键数据未就绪
  时停止新增信号，日报标记不完整，但持仓保护仍运行。
- 事件同步区分 `no_new_events`（接口成功但 0 行）与 `unavailable/failed`；
  F 在事件覆盖未确认时暂停，D 降档。
- 文档更新，避免把本地只读运行描述成自动同步成功。

**证据**：
- `test_data_update_reports_unavailable_without_faking_success`
- `test_event_sync_distinguishes_no_new_events_from_unavailable`
- `test_daily_stops_new_signals_but_keeps_protection_when_data_not_ready`
- `tests/test_jobs.py::test_daily_job_is_idempotent` 等幂等/重试测试。

## 9. 浏览器动态监控与日报

**根因**：SSE 只更新通知计数/任务标签；持仓价格、保护线和建议不刷新；
报告没有逐股行动建议。

**修复**：
- SSE 快照增加 `positions`（价格、保护线、动作、原因、建议数量、
  数据时间、估值状态）、`quote_freshness`、`valuation_incomplete`、
  `last_reconciled_at`；前端按 `data-position-code` 更新页面，风险
  toast 去重，可选声音，断线转轮询；连接状态与行情新鲜度分开显示。
- 持仓页新增动作/建议/受限/原因/数据时间列和估值不完整提示；通知页对
  历史提醒标注“以当前持仓决策为准”。
- 日报逐股列出持仓、策略、状态、建议数量、原因、保护线、风险、下一
  观察条件；页面/报告读取同一 `PositionDecisionService` 结果。
- 明确不承诺浏览器关闭或手机锁屏时送达。

**证据**：
- `tests/test_api.py::test_manual_execution_updates_positions`
- `test_sse_stream_emits_positions_and_freshness_after_login`
- `test_report_contains_position_actions_from_shared_decisions`

## 10. 验收与交付

- 全量测试、ruff、mypy 均通过（最终数字见 `test-report.md`）。
- 新增 27 个针对性回归测试，覆盖上述复现与边界。
- 未推送远端、未部署生产。全部变更保留在本地修复分支/工作区，供人工审阅。
- 仍属外部阻塞/部分完成的事项：
  - 真实 Tushare 权限、限频、公告全文/政策/新闻覆盖需要在有 token 的
    环境做合同级验证；当前测试使用注入 fake。
  - 实时报价在真实盘中的时延/可用性、券商撤单返回值对账未验证。
  - 历史除权除息与退市完整生命周期的回测精度受数据 vintage 限制。
  - 浏览器端仅通过 TestClient/接口契约验证，未做真实浏览器自动化；
    SSE/轮询逻辑需在目标浏览器复核。
  - 策略有效性未提升，仍为 `research`；软件修复不代表盈利。
