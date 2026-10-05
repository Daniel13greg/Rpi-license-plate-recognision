from pathlib import Path

import pytest
import yaml

from carwash_lpr.config import ConfigError, config_from_dict, expand_env, load_config, redacted

ROOT = Path(__file__).resolve().parent.parent


def minimal(**extra):
    return {"bays": [{"id": "1"}], **extra}


def test_defaults():
    cfg = config_from_dict(minimal(), environ={})
    bay = cfg.bays[0]
    assert bay.name == "Bay 1"
    assert bay.camera.type == "picamera2"
    assert bay.mode == "continuous"
    assert bay.roi == [0.0, 0.0, 1.0, 1.0]
    assert cfg.voting.min_reads == 2
    assert cfg.plates.accept == ["moldova", "moldova_special", "foreign"]
    assert cfg.webhook.url == ""
    assert cfg.device_id


def test_example_config_is_valid():
    env = {"CARWASH_API_TOKEN": "t", "CARWASH_HMAC_SECRET": "s", "LPR_API_TOKEN": "a"}
    cfg = load_config(ROOT / "config" / "config.example.yaml", environ=env)
    assert cfg.webhook.bearer_token == "t"
    assert len(cfg.bays) >= 1


def test_env_expansion():
    data = {"a": "${X}", "b": "${MISSING:-fallback}", "c": ["${X}/y"], "d": 5}
    assert expand_env(data, {"X": "1"}) == {"a": "1", "b": "fallback", "c": ["1/y"], "d": 5}
    with pytest.raises(ConfigError, match="MISSING"):
        expand_env("${MISSING}", {})


def test_types_are_coerced_from_strings():
    cfg = config_from_dict(minimal(api={"port": "${PORT}", "enabled": "yes"}), environ={"PORT": "9000"})
    assert cfg.api.port == 9000
    assert cfg.api.enabled is True


@pytest.mark.parametrize(
    "data, message",
    [
        (minimal(colour="red"), "unknown option"),
        ({"bays": [{"id": "1", "camera": {"typ": "opencv"}}]}, r"bays\[0\].camera: unknown option"),
        ({"bays": [{"id": "1", "camera": {"type": "usb"}}]}, "is not one of"),
        ({"bays": [{"id": "1", "roi": [0.5, 0, 0.4, 1]}]}, "roi"),
        ({"bays": [{"id": "1"}, {"id": "1", "camera": {"index": 1}}]}, "duplicate bay id"),
        ({"bays": [{"id": "1"}, {"id": "2"}]}, "same Raspberry Pi camera"),
        ({"bays": [{"id": "1"}, {"id": "2", "camera": {"type": "rpicam"}}]}, "same Raspberry Pi camera"),
        ({"bays": [{"id": "bay one"}]}, "letters, digits"),
        ({"bays": []}, "at least one bay"),
        ({"bays": [{"name": "x"}]}, r"bays\[0\].id: required"),
        (minimal(api={"port": 70000}), "outside the allowed range"),
        (minimal(api={"port": "eighty"}), "expected int"),
        (minimal(webhook={"url": "ftp://x"}), "http"),
        (minimal(plates={"accept": ["martian"]}), "is not one of"),
        ({"bays": [{"id": "1", "camera": {"type": "images"}}]}, "path: required"),
        (minimal(recognizer={"ocr_model_path": "x.onnx"}), "ocr_config_path"),
    ],
)
def test_invalid_configs(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data, environ={})


def test_relative_paths_follow_the_config_file(tmp_path):
    path = tmp_path / "conf" / "config.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump({"data_dir": "data", "bays": [{"id": "1", "camera": {"type": "images", "path": "frames"}}]}))
    cfg = load_config(path, environ={})
    assert cfg.data_dir == str(tmp_path / "conf" / "data")
    assert cfg.bays[0].camera.path == str(tmp_path / "conf" / "frames")


def test_load_errors(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("bays: [\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


def test_redacted_hides_secrets():
    cfg = config_from_dict(minimal(webhook={"url": "https://x", "bearer_token": "abc", "hmac_secret": ""}, api={"token": "t"}), environ={})
    shown = redacted(cfg)
    assert shown["webhook"]["bearer_token"] == "***"
    assert shown["webhook"]["hmac_secret"] == ""
    assert shown["api"]["token"] == "***"
