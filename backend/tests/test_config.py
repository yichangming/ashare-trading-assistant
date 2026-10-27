import pytest
from pydantic import ValidationError

from app.core.config import load_app_config, load_strategy_config


def test_load_strategy_config():
    config = load_strategy_config()
    assert config.name == "dual_template_v1"
    assert config.template_a.risk_pct == 0.025
    assert config.template_b.risk_pct == 0.02
    assert config.switching.ferment.b.max_positions == 2
    assert config.risk.max_positions == 3
    assert config.environment.stage_confirm_days == 2


def test_load_app_config():
    config = load_app_config()
    assert config.data.provider == "tushare"
    assert config.strategy_path.endswith("dual_template_v1.yaml")


def test_invalid_industry_weights_rejected(tmp_path):
    base = load_strategy_config()
    payload = base.model_dump()
    payload["industry"]["weights"]["return_20d"] = 0.9
    path = tmp_path / "bad.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_strategy_config(path)
