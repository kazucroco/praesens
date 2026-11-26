"""Settings: environment variables, live updates and the config file."""

import json

import pytest

from praesens_face.config import TUNABLE_FIELDS, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith("PRAESENS_"):
            monkeypatch.delenv(k)


def test_defaults_are_valid():
    s = Settings.from_env()
    assert s == Settings()
    assert s.review_threshold <= s.match_threshold


def test_env_values_are_coerced(monkeypatch):
    monkeypatch.setenv("PRAESENS_MATCH_THRESHOLD", "0.6")
    monkeypatch.setenv("PRAESENS_CONFIRM_FRAMES", "3")
    monkeypatch.setenv("PRAESENS_CTX_ID", "-1")
    monkeypatch.setenv("PRAESENS_CAMERA_SOURCE", "rtsp://cam/1")
    s = Settings.from_env()
    assert s.match_threshold == 0.6 and isinstance(s.match_threshold, float)
    assert s.confirm_frames == 3 and isinstance(s.confirm_frames, int)
    assert s.ctx_id == -1
    assert s.camera_source == "rtsp://cam/1"


@pytest.mark.parametrize("raw, expected", [
    ("1", True), ("true", True), ("Yes", True), (" on ", True),
    ("0", False), ("false", False), ("no", False), ("", False),
])
def test_env_bools(monkeypatch, raw, expected):
    monkeypatch.setenv("PRAESENS_CAMERA_AUTOSTART", raw)
    assert Settings.from_env().camera_autostart is expected


def test_bad_env_number_fails_loudly(monkeypatch):
    monkeypatch.setenv("PRAESENS_MATCH_THRESHOLD", "high")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_invalid_env_combination_fails(monkeypatch):
    monkeypatch.setenv("PRAESENS_REVIEW_THRESHOLD", "0.9")  # above the 0.45 default match
    with pytest.raises(ValueError, match="review_threshold"):
        Settings.from_env()


def test_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("PRAESENS_COOLDOWN_SECONDS", "5")
    assert Settings.from_env(cooldown_seconds=7.0).cooldown_seconds == 7.0


@pytest.mark.parametrize("changes", [
    {"match_threshold": 1.5},
    {"review_threshold": -2},
    {"review_threshold": 0.9},
    {"presence_seconds": -1},
    {"cooldown_seconds": -1},
    {"confirm_frames": 0},
    {"process_fps": 0},
])
def test_update_rejects_invalid_values_and_rolls_back(changes):
    s = Settings()
    before = s.tunable()
    with pytest.raises(ValueError):
        s.update(changes)
    assert s.tunable() == before


def test_update_rolls_back_every_field_in_a_failed_batch():
    s = Settings()
    with pytest.raises(ValueError):
        s.update({"cooldown_seconds": 1, "confirm_frames": 0})
    assert s.cooldown_seconds == Settings().cooldown_seconds


def test_update_refuses_fields_that_need_a_restart():
    s = Settings()
    with pytest.raises(ValueError, match="not tunable"):
        s.update({"model_name": "antelopev2"})
    assert s.model_name == "buffalo_l"


def test_tunable_lists_exactly_the_tunable_fields():
    assert set(Settings().tunable()) == TUNABLE_FIELDS


def test_save_without_config_file_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Settings().save()
    assert list(tmp_path.iterdir()) == []


def test_config_file_roundtrip(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    monkeypatch.setenv("PRAESENS_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("PRAESENS_MATCH_THRESHOLD", "0.55")
    s = Settings.from_env()
    assert s.match_threshold == 0.55
    s.update({"match_threshold": 0.6, "cooldown_seconds": 10})
    s.save()
    assert set(json.loads(cfg.read_text())) == TUNABLE_FIELDS
    s2 = Settings.from_env()
    assert (s2.match_threshold, s2.cooldown_seconds) == (0.6, 10)


def test_config_file_only_restores_tunable_fields(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"match_threshold": 0.5, "api_url": "http://evil/", "ctx_id": 3}))
    monkeypatch.setenv("PRAESENS_CONFIG_FILE", str(cfg))
    s = Settings.from_env()
    assert s.match_threshold == 0.5
    assert s.api_url == Settings().api_url and s.ctx_id == Settings().ctx_id


def test_missing_config_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAESENS_CONFIG_FILE", str(tmp_path / "not-yet.json"))
    assert Settings.from_env().match_threshold == Settings().match_threshold
