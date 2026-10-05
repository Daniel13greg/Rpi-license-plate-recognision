import json
import os

import pytest
import yaml

from carwash_lpr.cli import build_parser, load_env_file, main


def test_env_file_is_loaded_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / "env"
    env.write_text('# secrets\nCWL_A=plain\n\nCWL_B="quoted value"\nCWL_C=\'single\'\nCWL_D=keep-me\nnot a line\n')
    monkeypatch.setenv("CWL_D", "from-shell")
    try:
        assert load_env_file(str(env))
        assert (os.environ["CWL_A"], os.environ["CWL_B"], os.environ["CWL_C"]) == ("plain", "quoted value", "single")
        assert os.environ["CWL_D"] == "from-shell"
    finally:
        for key in ("CWL_A", "CWL_B", "CWL_C"):
            os.environ.pop(key, None)
    assert not load_env_file(str(tmp_path / "missing"))
    assert not load_env_file(str(tmp_path))  # a folder, not a file


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_unreadable_env_file_gives_a_hint(tmp_path, capsys):
    env = tmp_path / "env"
    env.write_text("CWL_X=1\n")
    env.chmod(0)
    assert not load_env_file(str(env))
    assert "sudo" in capsys.readouterr().err


def test_commands_see_the_service_secrets(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data_dir": str(tmp_path / "data"),
        "webhook": {"url": "${CWL_URL}", "bearer_token": "${CWL_TOKEN:-}"},
        "bays": [{"id": "1"}],
    }))
    env = tmp_path / "env"
    env.write_text("CWL_URL=https://carwash.example.md/lpr\nCWL_TOKEN=s3cret\n")
    try:
        assert main(["check-config", "-c", str(config), "--env-file", str(env)]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["webhook"]["url"] == "https://carwash.example.md/lpr"
        assert shown["webhook"]["bearer_token"] == "***"
    finally:
        os.environ.pop("CWL_URL", None)
        os.environ.pop("CWL_TOKEN", None)
    # without the env file the reference cannot be resolved
    assert main(["check-config", "-c", str(config), "--env-file", ""]) == 2
    assert "CWL_URL" in capsys.readouterr().err


def test_optional_config_defaults_to_the_installed_one(tmp_path, monkeypatch):
    assert build_parser().parse_args(["recognize", "car.jpg"]).config is None
    installed = tmp_path / "config.yaml"
    installed.write_text("bays: [{id: '1'}]\n")
    monkeypatch.setenv("CARWASH_LPR_CONFIG", str(installed))
    assert build_parser().parse_args(["recognize", "car.jpg"]).config == str(installed)
    assert build_parser().parse_args(["benchmark"]).config == str(installed)
