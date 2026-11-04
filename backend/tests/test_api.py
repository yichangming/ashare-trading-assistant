from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.db.session import reset_engine


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ASHARE_PASSWORD", "test-password")
    monkeypatch.setenv("ASHARE_SECRET_KEY", "test-secret-key")
    monkeypatch.delenv("ASHARE_HTTPS_ONLY", raising=False)
    reset_engine()
    application = create_app(database_url="sqlite://")
    with TestClient(application) as test_client:
        yield test_client
    reset_engine()


def _login(client: TestClient) -> None:
    response = client.post(
        "/login",
        data={"password": "test-password", "next": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_login_is_required_and_rejects_wrong_password(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")

    wrong = client.post(
        "/login",
        data={"password": "wrong", "next": "/"},
        follow_redirects=False,
    )
    assert wrong.status_code == 401
    assert "密码不正确" in wrong.text

    _login(client)
    assert client.get("/").status_code == 200


def test_json_api_requires_login_except_health(client: TestClient):
    assert client.get("/api/health").status_code == 200
    for path in (
        "/api/data/status",
        "/api/regime",
        "/api/candidates",
        "/api/plans",
        "/api/positions",
        "/api/executions",
        "/api/notifications",
        "/api/backtests",
        "/api/csv/template/positions.csv",
    ):
        assert client.get(path).status_code == 401, path
    assert client.get("/api/events/stream").status_code == 401


def test_all_eight_pages_render_after_login(client: TestClient):
    _login(client)
    for path in (
        "/",
        "/opportunities",
        "/positions",
        "/plans",
        "/reports",
        "/backtest",
        "/notifications",
        "/settings",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "A 股" in response.text or "交易辅助" in response.text, path


def test_demo_banner_is_explicit_and_switchable(client: TestClient):
    _login(client)
    assert "演示模式" not in client.get("/").text

    client.post("/settings/demo", data={"enabled": "1"})
    assert "演示模式" in client.get("/").text

    client.post("/settings/demo", data={"enabled": "0"})
    assert "演示模式" not in client.get("/").text


def test_csv_template_download(client: TestClient):
    _login(client)
    response = client.get("/api/csv/template/positions.csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert response.text.splitlines()[0].startswith("ts_code,")


def test_manual_execution_updates_positions(client: TestClient):
    _login(client)
    response = client.post(
        "/positions/execute",
        data={
            "ts_code": "000001.SZ",
            "side": "buy",
            "price": "10",
            "shares": "1000",
            "fees": "5",
            "executed_at": "2026-10-27T10:00:00",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    payload = client.get("/api/positions").json()
    positions = payload["positions"]
    assert positions[0]["ts_code"] == "000001.SZ"
    assert positions[0]["shares"] == 1000
    decisions = payload["decisions"]
    assert decisions[0]["action"] == "data_anomaly"
    assert decisions[0]["reason_code"] == "stop_missing"
    for field in (
        "ts_code",
        "primary_strategy",
        "sub_mode",
        "rule_version",
        "suggested_shares",
        "restricted_shares",
        "current_stop",
        "targets",
        "confirmed",
        "executed_shares",
        "pending_shares",
        "data_time",
    ):
        assert field in decisions[0]
    assert "valuation" in payload
    page = client.get("/positions")
    assert 'data-position-code="000001.SZ"' in page.text
    assert "数据异常" in page.text


def test_sse_stream_emits_positions_and_freshness_after_login(client: TestClient):
    import json

    _login(client)
    with client.stream("GET", "/api/events/stream") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        first = next(response.iter_lines())
    assert first.startswith("data:")
    payload = json.loads(first.removeprefix("data:").strip())
    assert "positions" in payload
    assert "quote_freshness" in payload
    assert "valuation_incomplete" in payload
    assert "last_reconciled_at" in payload


def test_plan_confirmation_is_persisted(client: TestClient):
    from datetime import date

    from app.db.repositories import PlanRepository
    from app.db.session import session_scope
    from tests.test_backtest import _plan

    _login(client)
    plan = _plan(date(2026, 11, 2))
    with session_scope() as session:
        PlanRepository.save(session, plan)
    response = client.post(f"/api/plans/{plan.id}/confirm")
    assert response.status_code == 200
    with session_scope() as session:
        assert "用户已确认计划" in PlanRepository.get(session, plan.id).plan_json["reasons"]


def test_api_sse_and_page_prefer_newer_intraday_decision(client: TestClient):
    import json
    from datetime import datetime, timedelta
    from decimal import Decimal
    from zoneinfo import ZoneInfo

    from app.db.repositories import PositionRepository
    from app.db.session import session_scope
    from app.domain.enums import PositionAction, ValuationStatus
    from app.domain.models import PositionDecision

    shanghai = ZoneInfo("Asia/Shanghai")
    now = datetime.now(shanghai)
    _login(client)
    with session_scope() as session:
        PositionRepository.upsert(
            session,
            account_id="default",
            ts_code="999999.SZ",
            shares=1000,
            available_shares=1000,
            cost_price=10,
            current_stop=9.5,
            current_price=10,
            primary_strategy="A",
            initial_r=0.5,
            initial_shares=1000,
        )
        position = PositionRepository.get(session, "default", "999999.SZ")
        decision = PositionDecision(
            ts_code="999999.SZ",
            action=PositionAction.EXIT,
            reason_code="hard_stop",
            explanation="盘中价格跌破保护线",
            trigger_price=Decimal("9.0"),
            data_time=now - timedelta(minutes=1),
            valid_until=now + timedelta(hours=1),
            current_stop=Decimal("9.5"),
            total_shares=1000,
            available_shares=1000,
            suggested_shares=1000,
            decision_source="intraday",
            valuation_status=ValuationStatus.OK,
        )
        position.last_decision_json = decision.model_dump(mode="json")
    payload = client.get("/api/positions").json()
    current = next(
        item for item in payload["decisions"] if item["ts_code"] == "999999.SZ"
    )
    assert current["action"] == "exit"
    assert current["decision_source"] == "intraday"
    assert float(current["trigger_price"]) == 9.0
    assert current["is_current"] is True

    page = client.get("/positions")
    assert page.status_code == 200
    assert 'data-position-code="999999.SZ"' in page.text
    assert "清仓" in page.text

    with client.stream("GET", "/api/events/stream") as response:
        assert response.status_code == 200
        first = next(response.iter_lines())
    snapshot = json.loads(first.removeprefix("data:").strip())
    sse_position = next(
        item for item in snapshot["positions"] if item["ts_code"] == "999999.SZ"
    )
    assert sse_position["action"] == "exit"
    assert sse_position["decision_source"] == "intraday"
    assert float(sse_position["current_price"]) == 9.0


def test_reports_page_renders_markdown_as_html(client: TestClient, tmp_path, monkeypatch):
    from app.api import app as app_module

    _login(client)
    report_dir = tmp_path / "data" / "reports" / "2026-11-03"
    report_dir.mkdir(parents=True)
    (report_dir / "daily.md").write_text(
        "# 日报\n\n## 今日摘要\n\n- 市场：防守\n\n"
        "| 策略 | 权限 |\n|---|---|\n| A 趋势回踩 | 暂停 |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "BACKEND_ROOT", tmp_path)
    page = client.get("/reports?date=2026-11-03")
    assert page.status_code == 200
    assert "<h2>今日摘要</h2>" in page.text
    assert "<table>" in page.text
    assert "<li>市场：防守</li>" in page.text
    assert "## 今日摘要" not in page.text
    assert "SELECTED" not in page.text
