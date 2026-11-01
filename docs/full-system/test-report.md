# 测试报告 — 完整 A–F 系统

> 日期：2026-11-01
> 分支：`full-system-v2`
> 环境：Linux，Python 3.12，uv；测试使用内存 SQLite 与注入的假 Tushare 客户端。
> 原则：**软件测试通过 ≠ 策略盈利**。demo/回放绩效不作为策略有效性证据。

## 1. 执行结果

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元/契约/集成测试 | `cd backend && uv run pytest` | **842 个测试收集，全部通过** |
| 静态检查 | `uv run ruff check app tests` | 通过（0 error） |
| 类型检查 | `uv run mypy app` | 通过（0 error；无 stubs 的第三方包已忽略） |
| 本机真实数据 smoke | `uv run python -m app.cli daily --day 2026-10-27` | `status=succeeded`，日报/快照/任务/通知落库 |
| CLI 注册 | `uv run python -m app.cli --help` | 旧数据命令 + `serve/migrate/daily/backtest/events-import` |
| 迁移 | `uv run alembic upgrade head` | 17 张表（16 业务表 + `alembic_version`），可重复执行 |

Smoke 使用本机已同步的 2026-10-27 日线数据，不调用外部接口；结果写入临时
SQLite，未污染正式账户库。当日市场为 defensive/narrow，系统按规则将 A/B/C
暂停、D/E/F 降档，只输出观察候选，未生成计划——这是权限逻辑的预期行为。

## 2. 覆盖范围（对应验收 A1–A19）

| 验收项 | 覆盖 | 主要测试 |
|---|---|---|
| A1 A–F 真实规则 | 部分/持续 | `test_strategies_daily.py`, `test_b2_intraday.py`；A 浅/深、B1、C1/C2、D、E、F 有样例；黄金样例仍需扩充 |
| A2 子模式独立 | 部分 | C1/C2、B1/B2、F 四类事件子模式；D uptrend/downtrend 与 A 浅/深部分覆盖 |
| A3 六模块贯通 | 是 | `test_full_system.py`, `test_pipeline.py`, `test_api.py` |
| A4 状态机 | 部分 | 计划过期/取消/成交/部分成交/平仓；`OBSERVING/PENDING_TRIGGER` 主要存在于快照，未全部落为计划行 |
| A5 人工成交持仓 | 是 | `test_portfolio.py`：T+1、部分卖出、费用、CSV、导入后卖出 |
| A6 加仓规则 | 是 | `test_full_system.py`：最多 2 次、≤首次 50%、不降保护线、B/D/E 禁止 |
| A7 止损/止盈优先级 | 部分 | 回测同日止损优先、盘中硬止损/接近止损、日线保护线/止盈/时间提示；卖出仍为人工 |
| A8 账户 CSV | 部分 | 五类模板、去重、成交、现金、公司行动、审计；订单 CSV 的失败撤单场景仍少 |
| A9 过期/风险未配置 | 是 | 风险未配置拒绝、真实账户 stale 拒绝、过期计划转 EXPIRED 且移出风险占用 |
| A10 回测与实盘同源 | 是 | `BacktestEngine` 调用同一 `StrategyPipeline`；`test_engine_reuses_pipeline_and_records_curve` |
| A11 指标/归因 | 是（有限） | 收益/年化/回撤/Sharpe/Sortino/Calmar/胜率/盈亏比/期望R/换手/成交率/持仓时间/资金占用/基准；策略与子模式归因；walk-forward/消融/成本敏感性 |
| A12 demo 标记 | 部分 | 页面 banner、回测 `is_demo`；没有独立 demo 数据集 |
| A13 Web 八页 | 是 | `test_api.py` 逐页 200 |
| A14 SSE/轮询 | 是 | SSE 首帧、连接状态、通知计数/任务状态；断线回退轮询 |
| A15 通知 | 是 | 风险优先排序、去重、确认/未处理、保留期清理 |
| A16 调度 | 是 | 交易日门控、幂等、单飞、失败持久化、重启恢复、指数退避重试、数据健康任务 |
| A17 安全 | 部分 | 登录、无默认密码、API/报告鉴权、127.0.0.1 默认、日志脱敏；会话为签名 Cookie 而非服务端会话表 |
| A18 部署 | 是 | Dockerfile/Compose、Alembic、备份脚本、恢复手册、HTTPS 说明 |
| A19 外部阻塞 | 是 | 数据源文档与页面明确标注实时/公告/券商限制，不伪造 |

## 3. 必测边界

已覆盖：

- 计划过期、部分成交、重复成交导入、撤单（基础路径）、入出金、公司行动。
- 账户未配置/过期、数据缺失降级、B2 未完成 bar、14:50 截止、10 分钟有效期。
- 涨停不可买、跌停不可卖、停牌无成交、跳空止损、同日止损优先。
- 多策略同股（第二策略拒绝）、行业/heat/现金/单票/相关性约束。
- 回测截断/复现基础、费用与滑点、B2 分钟回放缺失标注。
- 登录失败/成功、未登录 API 401、demo banner 切换、CSV 模板下载。

仍需真实数据或更多样例：

- 历史退市股完整生命周期、除权除息下的回测复权损益。
- 事件全文/政策/合同的人工核验流程与四类事件长期统计。
- 实时报价在真实盘中的时延/可用性；`stk_mins` 并发/限频下的完整轮询。
- 券商成交/撤单返回值与本地订单状态的长期对账。

## 4. 策略有效性（独立结论）

本报告**不**给出盈利结论。当前约束：

- 策略参数为 `research_defaults_v1`；没有达到样本外/滚动验证门槛。
- B2 依赖分钟数据，F 依赖人工核验事件；两者的长期样本有限。
- 回测结果受数据 vintage、停牌/涨跌停处理、费用假设影响。
- demo 与回放结果只能证明软件行为，不能证明 edge。

在满足每个策略/子模式的样本外、rolling、成本敏感性、环境分层和纸面跟踪
要求前，策略状态应保持 `research` / `paper`，不得标记为 `live_candidate`。
