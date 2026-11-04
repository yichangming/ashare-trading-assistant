# 测试报告 — 完整 A–F 系统（含 2026-11-02 修复回归）

> 日期：2026-11-02
> 分支：`fix/full-system-review`（基于审查基线 `0edeb3d`，未推送远端）
> 环境：Linux，Python 3.12，uv；测试使用内存 SQLite 与注入的假 Tushare /
> 分钟 / 报价客户端。
> 原则：**软件测试通过 ≠ 策略盈利**。demo/回放绩效不作为策略有效性证据。

## 1. 执行结果

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元/契约/集成测试 | `cd backend && uv run pytest` | **896 个测试，全部通过** |
| 静态检查 | `uv run ruff check app tests` | 通过（0 error） |
| 类型检查 | `uv run mypy app` | 通过（0 error；无 stubs 的第三方包已忽略） |
| 本机真实数据 smoke | `uv run python -m app.cli daily --day 2026-10-27` | 见下方说明；本机数据可运行，无 token 时同步阶段明确 unavailable |
| CLI 注册 | `uv run python -m app.cli --help` | 旧数据命令 + `serve/migrate/daily/backtest/events-import` |
| 迁移 | `uv run alembic upgrade head` | 基线 17 张表 + 本轮新增列；可重复执行 |

原有测试中与修复后契约直接冲突的断言已同步更新，而非放宽标准：
实际成交默认审计式录入（旧测试改用 `strict_plan=True` 验证拒绝路径）、
Tushare `vol` 按“手”换算股数、B2 观察池改为上一交易日快照、报告生成器
接受新的决策/数据说明参数、API 成交日期与本地最新处理日一致。

本轮新增 `tests/test_review_fixes.py`（27 项）以及迁移/API 回归，覆盖：
统一持仓决策、B 保护线复现、无计划待补全、分笔成交与真正加仓、计划偏离
审计、估值/入出金、日线顺序、B2 跨周末/午休/重启/多信号/限频、盘中旧报价
/缺时间/未来时间/去重/恢复/休市、未成交计划保留、费用对账、撤单状态、
全链路、报告、用户最大回撤。

Smoke 使用本机已同步的 2026-10-27 日线数据；无外部 token 时自动更新阶段
显示 unavailable，不会伪装同步成功。若当日数据/权限导致 A/B/C 暂停，只
输出观察候选是权限逻辑的预期行为。

另做了一次真实 HTTP 进程 smoke（临时数据库，`ASHARE_ENABLE_SCHEDULER=0`）：
未登录 `/api/positions` 返回 401；登录页 200、登录 POST 303；总览/持仓/
报告页 200；录入成交 303；`/api/positions` 返回现金、持仓市值、权益和
`decisions`（无计划成交显示 `data_anomaly/stop_missing`）；SSE 首帧包含
持仓、行情新鲜度和估值字段。该 smoke 不执行浏览器 JavaScript/声音/断线
重连，真机浏览器验证仍待执行。

## 2. 覆盖范围（对应验收 A1–A19）

| 验收项 | 覆盖 | 主要测试 |
|---|---|---|
| A1 A–F 真实规则 | 部分/持续 | `test_strategies_daily.py`, `test_b2_intraday.py`；A 浅/深、B1、C1/C2、D、E、F 有样例；黄金样例仍需扩充 |
| A2 子模式独立 | 部分 | C1/C2、B1/B2、F 四类事件子模式；D uptrend/downtrend 与 A 浅/深部分覆盖 |
| A3 六模块贯通 | 是 | `test_full_system.py`, `test_pipeline.py`, `test_api.py`, `test_review_fixes.py::test_full_chain_plan_partial_fills_valuation_reduce_and_close` |
| A4 状态机 | 是（计划/订单） | 计划过期/取消/成交/部分成交/平仓；回测订单区分未触发/部分/成交/过期/结构失效/用户取消/撤单未确认 |
| A5 人工成交持仓 | 是 | `test_portfolio.py`：T+1、部分卖出、费用、CSV、导入后卖出、分笔首次买入、加仓/减仓/清仓/重开 |
| A6 加仓规则 | 是 | `test_review_fixes.py`：首次多笔不算加仓、最多 2 次、≤首次 50%、不降保护线、B/D/E 禁止、卖出不重置计数 |
| A7 止损/止盈优先级 | 是（共享规则） | 回测与实盘共用 `evaluate_exit`；同日止损优先、硬止损、保护线上移、分批止盈、时间/事件退出；新保护线不倒用于当日低点 |
| A8 账户 CSV | 部分/是 | 五类模板、去重、成交、现金、公司行动、审计；订单 CSV 的撤单失败场景仍少 |
| A9 过期/风险未配置 | 是 | 风险未配置拒绝、真实账户 stale/估值不完整拒绝、过期计划转 EXPIRED 且移出风险占用 |
| A10 回测与实盘同源 | 是 | `BacktestEngine` 调用同一 `StrategyPipeline`；退出调用同一 `exit_rules`；`test_live_and_backtest_use_same_exit_rule_without_future_stop` |
| A11 指标/归因 | 是（有限） | 收益/年化/回撤/Sharpe/Sortino/Calmar/胜率/盈亏比/期望R/换手/成交率/持仓时间/资金占用/基准；策略与子模式归因；walk-forward/消融/成本敏感性；费用与权益对账 |
| A12 demo 标记 | 部分 | 页面 banner、回测 `is_demo`；没有独立 demo 数据集 |
| A13 Web 八页 | 是 | `test_api.py` 逐页 200；持仓页含动态决策字段 |
| A14 SSE/轮询 | 是 | `test_api.py`：SSE 含持仓、行情新鲜度、估值/对账时间；断线回退轮询 |
| A15 通知 | 是 | 风险优先排序、状态变化去重、确认/未处理、保留期清理、恢复记录 |
| A16 调度 | 是 | 交易日门控、幂等、单飞、失败持久化、重启恢复、指数退避重试、数据健康任务、自动数据更新阶段 |
| A17 安全 | 部分 | 登录、无默认密码、API/报告鉴权、127.0.0.1 默认、日志脱敏；会话为签名 Cookie 而非服务端会话表 |
| A18 部署 | 是 | Dockerfile/Compose、Alembic、备份脚本、恢复手册、HTTPS 说明 |
| A19 外部阻塞 | 是 | 数据源文档与页面明确标注实时/公告/券商限制，不伪造 |

## 3. 本轮缺陷修复验证

| 问题 | 状态 | 验证 |
|---|---|---|
| I 统一持仓行动决策 | 已修复 | `test_b_position_...`, `test_missing_plan_...`, API/SSE/报告测试 |
| II 回测/实盘退出不一致 | 已修复 | 共享 `ExitRules`/`evaluate_exit`；B 复现、无未来函数、止盈剩余 |
| III 成交归属/初始 R/加仓计数 | 已修复 | 分笔首次买入、真正两次加仓、部分卖出、重开、迁移待核对 |
| IV 账户估值/日线顺序 | 已修复 | 102,000 权益复现、缺行情不清零、入出金、旧 account 不覆盖 |
| V B2 观察池/生命周期 | 已修复 | 周五→周一、午休 11:25→13:05、重启、多信号资金竞争、全局限频 |
| VI 实时行情新鲜度/中断 | 已修复 | 2020 旧报价不触发止损、缺时间/未来时间 unknown、单只失败继续、去重/恢复/休市 |
| VII 回测未成交计划 | 已修复 | 1/2 计划→1/3 未成交→1/4 成交；费用对账；撤单状态 |
| VIII 自动数据更新 | 已实现/部分验证 | 阶段状态、无 token 不伪装、事件 no_new_events/failed、日线不完整仍保护持仓 |
| IX 浏览器动态/日报 | 已实现 | SSE 字段、持仓页动态属性、报告逐股建议；未做真实浏览器自动化 |
| X 验收/交付 | 完成软件部分 | 870 测试、ruff、mypy；外部数据/券商/浏览器真机为后续 |

逐项根因、修复位置和证据见 `docs/full-system/review-fixes.md`。

## 4. 必测边界

已覆盖：

- 计划过期、部分成交、重复成交导入、撤单（基础路径/未确认）、入出金、公司行动。
- 账户未配置/过期/估值不完整、数据缺失降级、B2 未完成 bar、14:50 截止、10 分钟有效期跨午休。
- 涨停不可买、跌停不可卖、停牌无成交、跳空止损、同日止损优先。
- 多策略同股（第二策略拒绝）、行业/heat/现金/单票/相关性约束。
- 回测截断/复现基础、费用与滑点、B2 分钟回放缺失标注、未成交订单保留。
- 登录失败/成功、未登录 API 401、demo banner 切换、CSV 模板下载、SSE 持仓字段。
- 旧日期/缺时间/未来时间报价、第一只失败第二只成功、重复告警、恢复、休市日。
- 周五观察池→周一、节假日、同日重复轮询、重启、午休有效期、多股票限频、多信号资金竞争。
- 分笔成交、真正两次加仓、部分卖出、再次加仓、全平重开、旧记录迁移待核对。

仍需真实数据或更多样例：

- 历史退市股完整生命周期、除权除息下的回测复权损益。
- 事件全文/政策/合同的人工核验流程与四类事件长期统计。
- 实时报价在真实盘中的时延/可用性；`stk_mins` 并发/限频下的完整轮询。
- 券商成交/撤单返回值与本地订单状态的长期对账。
- 真实浏览器中的 SSE 断线恢复、声音权限和页面长期运行。

## 5. 策略有效性（独立结论）

本报告**不**给出盈利结论。当前约束：

- 策略参数为 `research_defaults_v1`；没有达到样本外/滚动验证门槛。
- B2 依赖分钟数据，F 依赖人工核验事件；两者的长期样本有限。
- 回测结果受数据 vintage、停牌/涨跌停处理、费用假设、公司行动/退市
  覆盖影响；notes 会标注限制。
- demo 与回放结果只能证明软件行为，不能证明 edge。

在满足每个策略/子模式的样本外、rolling、成本敏感性、环境分层和纸面跟踪
要求前，策略状态应保持 `research` / `paper`，不得标记为 `live_candidate`。

## 6. 第二轮验收（2026-11-02，基线 97afc8a）

### 6.1 检查结果

| 检查 | 命令 | 结果 |
|---|---|---|
| 全量测试 | `cd backend && .venv/bin/pytest -q` | 退出码 0；`--collect-only` 共 **896** 项 |
| 静态检查 | `.venv/bin/ruff check app tests` | 通过（0 error） |
| 类型检查 | `.venv/bin/mypy app` | 通过（0 error） |
| 真实数据更新 + 盘后分析 | `.venv/bin/python -m app.cli daily --day 2026-11-02 --force` | `status=succeeded skipped=False plans=0` |
| 幂等重跑 | 同一命令去掉 `--force` | `status=succeeded skipped=True` |

### 6.2 数据与日历覆盖

- 处理后日线：2018-01-02 至 **2026-11-02**，共 2138 个交易日；最新日
  **5571** 只股票、5571 行，列包含 OHLCV、pre_close、amount、adj_factor、
  circ_mv、涨跌停价和停牌标记。
- 交易日历：2018-01-01 至 **2026-12-31**，覆盖 2026-11-02 及未来计划
  有效期；2026-11-02 为开市日。
- 参考数据：stock_basic 5920 行、namechange 14252 行、
  index_member 7558 行、index_classify 28 行。
- 本轮增量同步：2026-10-27、10-28、10-29、10-30、11-02 共 5 个交易日；
  组装 5571 行；质量门通过；结构化事件同步首次 `ok`
  （repurchase 287、stk_holdertrade 346、forecast_vip 191、
  express_vip 11、income_vip 5588），重跑为 `no_new_events`（接口成功但
  0 行新事件，不等于覆盖缺失）。数据更新总状态 `ready`。

### 6.3 盘后分析结果（2026-11-02）

- 市场：趋势 `defensive`，情绪 `start`，风险系数 0.00，宽度 18.70%
  （narrow），波动 normal，流动性 normal，全局风险开关 normal，无数据降级。
- 策略权限：A/B/C `paused`（market conditions not permitted）；D
  `reduced`（0.50）；E `normal`（1.00）；F `reduced`（0.50）。
- 策略漏斗：A/B/C 候选 0（权限暂停）；D 候选 2（观察 1、确认 1，
  301047.SZ 因 D 止损距离 16.14% > 8.00% 被计划拒绝）；E 候选 4
  （全部观察）；F 候选 0（需要 C1/C2 候选与已核验正面事件，本次无符合项，
  未降低标准）。
- 计划：0 个；没有候选时报告解释权限/策略门槛，不强行选股。
- 持仓：空账户，无持仓决策；账户权益 1,000,000，无估值不完整。
- 报告：`backend/data/reports/2026-11-02/daily.md|html`。
- 网页/API：登录后 `/`、`/opportunities`、`/reports`、`/positions`、
  `/settings` 均 200；`/api/candidates` 返回同一 6 个候选/5 个观察；
  `/api/data/status` 最新处理日 2026-11-02。

### 6.4 专项修复验证

| 问题 | 状态 | 主要验证 |
|---|---|---|
| 页面/API/SSE 使用最新有效决策 | 已修复 | `test_current_decisions_prefer...`、`test_api_sse_and_page_prefer...` |
| 盘中估值同步 | 已修复 | `test_intraday_revalues_before_b2...` |
| 保护线生效时点/幂等 | 已修复 | `test_daily_stop_is_effective_next_session...` |
| 日历覆盖与刷新 | 已修复 | `test_data_update_tries_calendar_refresh...`、`...marks_unknown...` |
| B/D 时间退出 | 已修复 | `test_no_new_high_uses_trading_days...`、`test_service_records...` |
| 报价新鲜度统一 | 已修复 | `test_same_day_expired_quote...` |
| 部分卖单不掩盖风险 | 已修复 | `test_small_pending_sell_order...` |
| 计划取消/撤单确认分离 | 已修复 | `test_plan_cancel_and_order_cancel_keep_exposure...` |
| 回测事件/MA5/区间/ATR | 已修复 | `test_backtest_f_event_expiry...`、`test_backtest_ma5_range...` |
| 缺行情/缺止损优先级 | 已修复 | `test_stop_missing_has_priority...`、`test_manual_execution_updates_positions` |

### 6.5 研究选股与账户限制

在临时数据库中用 `AccountMode.REAL` 且 `risk_config_configured=False`
运行同一 2026-11-02 流水线：仍生成 E 策略 4 个观察候选，计划数为 0；
账户风险配置缺失只阻止真实数量建议（既有
`test_full_system.py::test_real_account_without_risk_configuration_is_rejected`），
不隐藏研究结果。

### 6.6 本轮外部限制

- 真实 Tushare 日线/参考/结构化事件接口已在本环境验证；公告全文、政策、
  新闻等无稳定接口的覆盖仍需人工核验。
- B2 分钟级盘中监控未在本次盘后验收中执行真实盘中轮询。
- 未接券商下单/撤单接口；委托成交/撤单确认依赖导入或人工对账。
- 浏览器仅验证 TestClient/HTML/SSE 契约，未做真实浏览器自动化。
- 策略有效性未提升，仍为 research/paper。

### 6.7 日报中文化（2026-11-04）

- 新增 `app/jobs/report_text.py`：策略/子模式/趋势/情绪/权限/估值/动作/
  原因/拒绝原因/数据降级的中文映射，以及报告用 Markdown → HTML 渲染器。
- 日报顶部新增“今日摘要”，说明市场状态、是否生成新计划、持仓监控是否
  运行以及持仓操作提醒；未运行持仓监控时不会显示“无操作提醒”。
- 风险系数明确解释为“风险预算乘数”，不是持仓比例；策略权限、情绪阶段
  `start`、数据降级 `limit_list_d` 等均按代码实际含义用中文说明。
- 计划拒绝原因改为完整中文句子，保留股票代码、实际数值和阈值。例如
  2026-11-03：`D 超跌修复策略有 1 只股票通过策略确认，但 301563.SZ 的
  拟入场价到止损价的距离为 10.64%，超过 8.00% 的上限，未生成交易计划。`
- `/reports` 页面和独立 `daily.html` 均直接渲染标题、列表和表格，不再
  在页面中显示 `#`、`##`、`>`；表格在窄屏下可横向滚动。
- 2026-11-03 重跑回归：策略权限 A/B/C paused、D reduced 0.5、E normal、
  F reduced 0.5；候选 10（D 6、E 4）、观察 9、确认 1、计划拒绝 1
  （301563.SZ 10.64% > 8.00%），与修改前一致。
- 全量测试 896 项通过，ruff/mypy 通过。
