# 完整 A-F 交易辅助系统 — 实施计划

> 基于 `docs/full-system/` 的架构/规则/参数/验收文档。
> 复用现有 M1 数据层、特征、点时股票池、行情判断、行业/龙头、仓位/风控；扩展 A-F、账户、回测、调度、Web、事件。
> 每个阶段产出可运行代码 + 测试；不接真实交易接口；不自动推送远端。

## 阶段

### P0 工程与配置
- 新增依赖：fastapi, uvicorn, sqlalchemy, alembic, jinja2, apscheduler, python-multipart, pwdlib[argon2]（或 scrypt）。
- `full_system_v1.yaml` 策略/权限/风险/费用/回测/调度配置。
- 配置模型 + 加载 + 测试。
- 分支 `full-system-v2`；本地提交。

### P1 领域与数据库
- 扩展 enums：Strategy A-F、SubMode、PlanStatus 完整状态、PermissionMode、AccountMode、EventType、VerificationStatus、JobStatus、NotificationLevel、ExecutionSide。
- 扩展 Candidate/TradePlan（向后兼容默认字段）；新增 StrategyPermission、Account、PositionLot、Execution、OrderIntent、CashFlow、CorporateAction、EventRecord、Notification、JobRun、AuditLog、BacktestRun。
- SQLAlchemy 模型、session、仓储、Alembic 初始迁移；SQLite `data/app.db`。
- 测试：模型校验、状态迁移、仓储 CRUD、迁移。

### P2 六类策略
- `strategies/base.py`：StrategyContext/协议/registry。
- A trend pullback（浅/深）；B1/B2；C1/C2；D uptrend/downtrend；E；F event→C1/C2。
- 特征/形态计算，严格无未来函数；B2 5 分钟状态机（止跌/冻结/确认/重置/10 分钟/14:50 边界/VWAP）。
- 每个策略/子模式独立测试 + 黄金样例 + 截断测试。
- 候选包含策略证据、数据时间、理由、失效条件。

### P3 行情权限与统一计划
- 四维行情判断 + 分策略 normal/reduced/paused + 全局风险/数据开关；2 日确认/滞回。
- 统一买点/止损/仓位/止盈按 A-F 分派；P/max_acceptable/执行区间/有效期/目标/R:R。
- 权益/仓位/行业/相关性/heat/现金/待成交/部分成交；加仓规则；压力情景。
- 测试：每个策略从候选到计划；风控拒绝/降档；T+1/整手/价格精度。

### P4 账户与成交
- 账户初始化、CSV 模板（positions/executions/orders/cash_flows/corporate_actions）、导入去重、部分成交、费用、撤单、出入金、公司行动、审计、过期检查。
- 持仓聚合/策略来源/主策略/可卖数量/加仓批次；卖出确认才释放风险。
- 测试：重复导入、部分成交、T+1、加仓、减仓、清仓、公司行动、过期。

### P5 事件
- 事件表/版本/撤回/去重；Tushare forecast_vip/express_vip/income_vip/repurchase/stk_holdertrade 入库；手动 CSV/JSON；人工核验；可选 AI 提取接口（不核心）。
- F 策略接入；5 日窗口；负面/撤回；同事件/股票去重。
- 测试：修订/撤回/重复/未核验/负面。

### P6 回测与模拟
- 日线回测：T+1、涨跌停/停牌/退市/跳空、费用/滑点、部分成交、同日止损优先；A-F 独立与组合。
- 5 分钟 B2 回放（仅可用数据）；缺失明确标注。
- 指标、归因、walk-forward、消融、成本敏感性；demo 数据。
- 测试：不变量、前视截断、退市、费用、复现。

### P7 调度、报告、通知
- APScheduler：17:00 盘后、盘中监控、数据健康、幂等/单飞/重试/重启恢复。
- 日报 HTML/Markdown；通知中心（风险优先/去重/确认）；SSE。
- 测试：重复运行、锁、重启、数据缺失、通知去重。

### P8 API/Web
- FastAPI 路由 + 单用户认证（scrypt/Argon2，无默认密码）+ 会话 + 127.0.0.1。
- 8 页中文响应式：总览/机会/持仓/计划成交/报告/回测/通知/设置健康。
- SSE 任务/通知；连接/数据时间；demo 切换；手工成交/CSV。
- 测试：TestClient 契约、登录、关键页面、SSE、demo/real 区分。

### P9 部署与交付
- Docker Compose/等价、Alembic、备份恢复、HTTPS/认证说明、日志脱敏。
- README/部署手册、参数表、数据接入、测试报告、外部权限与成本、已知限制。
- 全量测试 + 真实数据 smoke + demo 验收；本地提交。

## 约束

- 不自动下单；不接券商交易 API。
- 不推倒重写已有 M1/M2；旧 A/B 双模板测试保持通过。
- 外部阻塞：实时报价盘中验证、公告全文/新闻/政策、券商 API、分钟全历史；实现适配器/导入/演示并明确标注。
- 软件验收与策略盈利分开报告。
