# 部署手册（Docker Compose / systemd / 本地）

> 单用户、单机、人工执行。默认只监听 `127.0.0.1`；不接券商下单接口。
> 软件验收与策略盈利分开报告。

## 1. 前置条件

- 操作系统：Linux（示例为 Ubuntu/Debian）；Docker 24+ 与 Compose v2（推荐），或
  Python 3.12 + [uv](https://docs.astral.sh/uv/)。
- Tushare Pro token（`TUSHARE_TOKEN`），权限见 `docs/full-system/data-sources.md`。
- 一个登录密码与稳定的会话签名密钥。

## 2. 环境变量

在仓库根目录创建 `.env`（gitignored，模板见 `.env.example`）：

```text
TUSHARE_TOKEN=your_tushare_pro_token
ASHARE_PASSWORD=choose-a-long-password
ASHARE_SECRET_KEY=<python -c "import secrets; print(secrets.token_urlsafe(32))">
ASHARE_HTTPS_ONLY=0          # 仅当直接以 HTTPS 提供服务时设为 1
```

说明：

- 没有默认密码。缺少 `ASHARE_PASSWORD` / `ASHARE_PASSWORD_HASH` 时应用拒绝启动。
- 可用预哈希密码替代明文：`ASHARE_PASSWORD_HASH=scrypt$<salt>$<digest>`。
- `ASHARE_SECRET_KEY` 必须稳定；随机值会导致重启后所有会话失效。
- `TUSHARE_TOKEN` 缺失时系统仍可读取本地数据并做持仓保护，但自动同步/
  事件阶段会明确显示 `unavailable`，不会伪装成同步成功。
- `.env`、`backend/data/`、`*.db` 不进入版本库。

## 3. Docker Compose（推荐）

```bash
cp .env.example .env
# 编辑 .env（见上）
docker compose build
docker compose up -d
docker compose logs -f app
```

- 容器启动时先执行 `alembic upgrade head`，再以 **单 worker** 启动
  `uvicorn --factory app.api.app:create_app`。
- Compose 设置 `ASHARE_ENABLE_SCHEDULER=1`，因此 17:00 盘后任务、16:30 数据
  健康检查、盘中监控都会运行。盘后任务按“交易日检查→参考/日历→增量日线
  →组装→质量→事件→持仓估值/决策→策略/计划→日报”执行。**不要增加 worker
  数**：进程内 APScheduler 会被复制，任务锁也按单进程设计。无 token 时
  同步阶段标记 unavailable，新增信号停止但已有持仓保护继续。
- 端口映射为 `127.0.0.1:8000:8000`，默认不暴露到局域网。
- 持久化：`./backend/data` 挂载到 `/app/backend/data`，包含行情、SQLite
  `app.db`、报告、导入文件与备份。
- 健康检查：`GET /api/health` 返回 `status=ok|degraded` 与数据健康明细。

常用命令：

```bash
docker compose ps
docker compose exec app alembic current
docker compose exec app alembic upgrade head
docker compose exec app python scripts/backup_db.py --out data/backups --retain 30
docker compose down                 # 停止；数据卷仍在
```

## 4. 非 Docker：uv + systemd

```bash
cd backend
uv sync --frozen --no-dev
uv run alembic upgrade head
uv run python -m app.cli serve --scheduler
```

示例 systemd 单元 `/etc/systemd/system/ashare.service`：

```ini
[Unit]
Description=A-share trading assistant
After=network-online.target

[Service]
Type=simple
User=ashare
WorkingDirectory=/opt/stock/backend
EnvironmentFile=/opt/stock/.env
Environment=ASHARE_ENABLE_SCHEDULER=1
ExecStart=/opt/stock/backend/.venv/bin/uvicorn --factory app.api.app:create_app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ashare
journalctl -u ashare -f
```

## 5. HTTPS 与反向代理

默认仅本机 HTTP。需要局域网/远程访问时，在 Caddy/Nginx 终止 TLS，并把
`ASHARE_HTTPS_ONLY=1`，同时保持应用只监听 `127.0.0.1`：

```caddy
trading.example.com {
    reverse_proxy 127.0.0.1:8000
    encode gzip
}
```

- 反向代理必须透传 `X-Forwarded-Proto`，并限制来源 IP/VPN。
- 不要直接把 8000 端口暴露到公网；单用户 cookie 会话不是多用户认证系统。
- 打开 HTTPS 后，浏览器会话 Cookie 才会带 `Secure`。

## 6. 数据库迁移与备份恢复

迁移：

```bash
cd backend
uv run alembic current
uv run alembic upgrade head
```

`create_app` / `cli serve` / `cli daily` 也会调用迁移准备逻辑：内存库用
`create_all`，文件库走 Alembic；早期由 `create_all` 创建且无
`alembic_version` 的库会先 stamp 到 head，再正常升级。

在线备份（不停止服务，使用 SQLite backup API）：

```bash
cd backend
uv run python scripts/backup_db.py --database data/app.db \
  --out data/backups --retain 30
```

建议每日盘后任务完成后由 cron 调用一次：

```cron
30 17 * * 1-5 cd /opt/stock/backend && /opt/stock/backend/.venv/bin/python scripts/backup_db.py --out data/backups --retain 30
```

恢复：

1. 停止服务（`docker compose stop app` 或 `systemctl stop ashare`）。
2. 保留当前库副本：`cp data/app.db data/app.db.before-restore`。
3. 用备份覆盖：`cp data/backups/app-YYYYmmdd-HHMMSS.db data/app.db`。
4. 启动服务，检查 `alembic current` 与 `GET /api/health`。
5. 在页面“设置与健康”核对账户/数据时间；必要时重新导入 CSV。

`backend/data/raw`、`processed`、`reports`、`imports` 不在 SQLite 内，备份
`app.db` 时建议同时归档这些目录（行情可重新同步，报告/导入不可重建）。

## 7. 运行检查清单

- `GET /api/health`：`status`、`data.ok`、`issues`、`scheduler`。
- `/settings`：数据库、预期交易日、最新处理日、行情估值时间、人工核对时间、
  估值完整性、调度器状态。
- `/api/data/status`：最近任务及错误。
- `/notifications`：风险优先、去重、确认。
- 日志：`docker compose logs` 或 `journalctl`；日志过滤器会遮蔽
  `token/password/secret/authorization/api_key` 等键值。

## 8. 安全边界

- 不自动下单、不接券商交易 API；所有买卖由人工在券商端完成。
- 单用户、单进程调度；多 worker 会造成重复任务。
- 真实账户未配置风险参数或账户数据过期时，不生成可执行股数/新计划。
- 外部数据（实时报价、分钟线、公告全文）不可用时明确标注降级，不用日线
  冒充实时。

## 9. 已知限制

- 会话为签名 Cookie，不是服务端会话表；单用户场景足够，但不是多租户方案。
- `stk_mins` 约 1 次/分钟，限频在进程内全局协调；B2 标的较多时按队列
  逐个请求，会看到明确延迟，不能并发突破接口总限频。
- 回测不能用日线伪造 B2 盘中路径；缺少分钟数据时只回放可用区间。
- 历史除权除息与退市完整生命周期受数据 vintage 限制，回测 notes 会标注，
  不输出伪完整绩效。
- 事件全文/政策/部分合同无稳定接口，需要人工结构化导入与核验；结构化
  接口成功但 0 行只表示“确认无新事件”，不等于公告全覆盖。
- 盘后估值不是人工对账；真实账户的 `last_reconciled_at` 需用户显式更新。
- 浏览器 SSE/轮询只服务打开的页面；关闭浏览器、挂起标签或锁屏时不承诺
  通知送达。
- Docker 镜像未内置 HTTPS 证书；TLS 由反向代理负责。
