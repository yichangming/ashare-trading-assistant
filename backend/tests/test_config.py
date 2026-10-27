from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.core.config import load_app_config, load_strategy_config


def _modified_strategy(tmp_path: Path, section: tuple[str, ...], key: str, value: object) -> Path:
    payload = load_strategy_config().model_dump()
    target = payload
    for part in section:
        target = target[part]
    target[key] = value
    path = tmp_path / "strategy.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_strategy_config():
    config = load_strategy_config()
    assert config.name == "dual_template_v1"
    assert config.template_a.risk_pct == 0.025
    assert config.template_b.risk_pct == 0.02
    assert config.template_a.atr_buffer == 0.5
    assert config.template_b.atr_buffer == 0.5
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

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_strategy_config(path)


def test_leader_weights_must_sum_to_one(tmp_path):
    path = _modified_strategy(tmp_path, ("leader", "weights"), "rs_20d", 0.9)
    with pytest.raises(ValidationError, match="leader weights must sum to 1.0"):
        load_strategy_config(path)


def test_score_weights_must_sum_to_one(tmp_path):
    path = _modified_strategy(tmp_path, ("template_a", "score"), "leader", 0.9)
    with pytest.raises(ValidationError, match="score weights must sum to 1.0"):
        load_strategy_config(path)


def test_nested_unknown_key_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("template_a",), "unexpected_option", True)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_strategy_config(path)


def test_missing_config_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        load_strategy_config(missing)


def test_non_mapping_yaml_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_strategy_config(path)


def test_negative_risk_factor_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("switching", "start"), "risk_factor", -0.1)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_strategy_config(path)


def test_negative_max_positions_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("risk",), "max_positions", -1)
    with pytest.raises(ValidationError, match="risk.max_positions"):
        load_strategy_config(path)


def test_reversed_stop_bounds_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("template_a",), "stop_min_pct", 0.09)
    with pytest.raises(ValidationError, match="stop_min_pct must be <= stop_max_pct"):
        load_strategy_config(path)


@pytest.mark.parametrize("section", ["template_a", "template_b"])
@pytest.mark.parametrize("days", [0, -1])
def test_non_positive_time_stop_days_rejected(
    tmp_path: Path, section: str, days: int
) -> None:
    path = _modified_strategy(tmp_path, (section,), "time_stop_days", days)
    with pytest.raises(ValidationError, match="greater than 0"):
        load_strategy_config(path)


def test_nan_industry_weight_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("industry", "weights"), "return_20d", float("nan"))
    with pytest.raises(ValidationError, match="finite number"):
        load_strategy_config(path)


def test_out_of_range_risk_factor_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("switching", "start"), "risk_factor", 1.5)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        load_strategy_config(path)


def test_stage_max_positions_exceeding_risk_rejected(tmp_path):
    path = _modified_strategy(tmp_path, ("switching", "ferment", "b"), "max_positions", 3)
    with pytest.raises(ValidationError, match="must be <= risk max"):
        load_strategy_config(path)


def test_disabled_stage_policy_requires_zero_positions(tmp_path):
    path = _modified_strategy(tmp_path, ("switching", "freeze", "a"), "max_positions", 1)
    with pytest.raises(ValidationError, match="must be 0 when disabled"):
        load_strategy_config(path)
