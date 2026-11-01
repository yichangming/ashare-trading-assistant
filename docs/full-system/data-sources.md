# 数据接入说明与外部能力清单

> 实测时间：2026-11-01（休市日）。所有“可用”指该 token 当前权限下接口返回非空/无权限错误；
> 实际生产使用前仍需在交易时段复核实时延迟、限频和数据质量。

## 1. Tushare 账户能力（实测）

| 接口 | 状态 | 本次探测 | 说明 |
|---|---|---|---|
| `trade_cal` | 可用 | 10 行/10 日 | 交易日历 |
| `stock_basic` | 可用 | L/D/P 5,919 行 | 含退市；当前快照可能缺合并/退市代码 |
| `daily` | 可用 | 5,570 行/日 | 未复权日线，盘后 |
| `adj_factor` | 可用 | 5,580 行/日 | 复权因子 |
| `daily_basic` | 可用 | 5,570 行/日 | 换手、流通市值等 |
| `stk_limit` | 可用 | 5,654 行/日 | 涨跌停价 |
| `suspend_d` | 可用 | 7 行/日 | 含 `suspend_timing` 盘中停牌 |
| `index_daily` | 可用 | 9,328 行/日 | 全指数；可按 ts_code 过滤 |
| `limit_list_d` | 可用 | 91 行/日（2026） | 涨跌停列表 |
| `namechange` | 可用 | 分段 14,250 行 | 5 年分段；单次无参数有 10,000 上限 |
| `index_classify` | 可用 | L1 28 行 | 申万一级 |
| `index_member` | 可用 | 按 L1 合计 7,558 行 | 按行业拉取；单次无参数 3,000 上限 |
| `stk_mins` | 可用 | 5 分钟 49 根/日 | **1 次/分钟**；1 分钟同接口受限 |
| `index_mins` | 不可用 | 接口名错误 | 指数分钟暂不可用 |
| `rt_k` | 无权限 | 权限错误 | 不使用 |
| `realtime_quote` | 休市返回空 | `None` | 免费源；盘中延迟/可用性未验证 |
| `forecast` / `forecast_vip` | 可用 | 按公告日/报告期 | 业绩预告 |
| `express` / `express_vip` | 可用 | 11 行/期 | 业绩快报 |
| `income` / `income_vip` | 可用 | 需 ts_code；VIP 按报告期 8,715 行 | 财报 |
| `repurchase` | 可用 | 793 行/月 | 回购 |
| `stk_holdertrade` | 可用 | 1,781 行/月 | 股东增减持 |
| `disclosure_date` | 可用 | 0 行（本次区间） | 披露计划 |
| `anns_d` | 无权限 | 权限错误 | 公告全文不可用 |
| `news` / `cctv_news` | 无权限 | 权限错误 | 新闻/政策不可用 |

## 2. 实时与分钟能力

### 2.1 5 分钟线（B2 核心）

- 接口：`stk_mins(ts_code, freq="5min", start_date, end_date)`。
- 返回：`trade_time, open, high, low, close, vol, amount`；49 根/完整交易日（含 09:35–15:00）。
- 限频：**1 次/分钟**；因此只能监控一个小观察池（持仓 + 前一交易日 B 候选）。
- 时点：只使用已完成 bar；不使用未来 bar。
- VWAP：`amount/vol` 与价格量级一致（实测 11.64 vs 11.645）；精确计算，异常时标记 `vwap_approximate`。
- 历史回补：1 次/分钟意味着全市场 5 年分钟回补不可行；只回补观察池/近期区间，回测缺失处明确标注。

### 2.2 实时报价

- `ts.realtime_quote` 在休市返回 `None`；盘中可用性与延迟**未验证**。
- 系统实现 `RealtimeQuoteAdapter`：
  - 若盘中可用：用于持仓保护线/硬风险监控和 B2 辅助；
  - 若不可用/延迟超阈值：回退到 5 分钟 bar（1 次/分钟）并标记 `realtime_degraded`；
  - 健康状态在 Web “运行健康”显示；不可用时不得声称实时。
- 不使用 `rt_k`（无权限）。

### 2.3 指数盘中

- `index_mins` 不可用；指数趋势用日线，盘中指数只作参考（如实时报价可用）。
- 市场宽度/情绪盘中只依据当时已知的日线/已完成分钟数据更新，并标注数据时间。

## 3. 事件数据

### 3.1 可自动入库

- 业绩预告/快报/财报：按 `ann_date`/`period` 拉取，保留 `first_ann_date`、`update_flag`、`summary`、`change_reason`；修订生成新版本。
- 回购：`ann_date`、`proc`（预案/实施/完成）、金额/价格区间/进度。
- 股东增减持：`ann_date`、`in_de`、`change_vol/ratio`、`avg_price`、`holder_type`。
- 财报：`income_vip` 按报告期；`express_vip` 按报告期。
- 入库幂等键：`(source, event_type, ts_code, ann_date, source_key)`；修订/撤回单独记录。

### 3.2 需人工/结构化导入

- 公告全文（`anns_d` 无权限）、政策、重大合同、中标、产品获批。
- 支持 CSV/JSON 模板：
  `event_type, ts_code(s), title, summary, source, source_url, published_at, stage, certainty, direction, materiality, verification_status, raw_payload`
- 人工核验后才可 `verified`；`pending_verification` 不进入真实资金。
- AI 提取为可选增强：输出结构化候选，必须人工核验；不得作为唯一依据。

### 3.3 缺失处理

- 事件数据缺失：F 权限 `paused/reduced`，报告标注 `event_data_incomplete`；D 的负面事件风险同样标注。
- 不得用“无事件”推断安全。

## 4. 数据存储与同步

- 原始：`backend/data/raw/<endpoint>/<YYYYMMDD>.parquet` + reference + manifests。
- 清洗：`backend/data/processed/{bars,index_bars,limit_list}/trade_date=...`。
- 分钟：`backend/data/raw/stk_mins/<YYYYMMDD>/<ts_code>.parquet`（观察池/近期）；不做全市场伪造。
- 事件：SQLite `events` 表 + 原始 JSON/CSV 归档（`data/events/`）。
- 账户/成交/计划/通知/任务/审计：SQLite `backend/data/app.db`。
- 数据版本：日 manifest 哈希；分钟/事件各自 snapshot id；计划保存 `data_version`。
- 交易日历/时区：`Asia/Shanghai`，`trade_cal` 为准。

## 5. 外部成本与阻塞

| 项目 | 成本 | 状态 |
|---|---|---|
| Tushare 日线/事件/分钟 | 由用户积分/权限决定 | 当前 token 可用；`stk_mins` 1 次/分钟 |
| 实时行情 | Tushare 免费源；稳定性未保证 | 休市无法验证 |
| 公告全文/新闻/政策 | 无权限 | 阻塞；人工/结构化导入 |
| 券商交易 API | v1 不接 | 阻塞（设计选择）；CSV/手工 |
| 微信/Server酱 | 不引入 | 站内通知 |
| 分钟全历史回补 | 受 1 次/分钟限制 | 不可行；只回补观察池/近期 |
