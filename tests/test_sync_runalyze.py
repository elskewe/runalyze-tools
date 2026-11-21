import datetime
import json
import os
import runpy
import stat
from pathlib import Path

import pytest


def load_sync_runalyze():
    script = Path(__file__).resolve().parents[1] / "sync-runalyze.py"
    return runpy.run_path(str(script))


def test_missing_sync_checkpoint_starts_at_utc_midnight(tmp_path):
    sync_runalyze = load_sync_runalyze()
    checkpoint = tmp_path / "sync-runalyze-config.txt"
    now = datetime.datetime(2026, 10, 1, 14, 30, tzinfo=datetime.timezone.utc)

    start_time = sync_runalyze["_load_sync_start_time"](str(checkpoint), now)

    assert start_time == datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
    assert checkpoint.read_text(encoding="utf-8") == "2026-10-01 00:00"


def test_missing_runalyze_credentials_are_created_from_prompt(monkeypatch, tmp_path):
    sync_runalyze = load_sync_runalyze()
    credentials_file = tmp_path / "runalyze_credentials.json"
    monkeypatch.setitem(
        sync_runalyze["_load_runalyze_credentials"].__globals__,
        "getpass",
        lambda _prompt: "test-token",
    )

    credentials = sync_runalyze["_load_runalyze_credentials"](str(credentials_file))

    assert credentials == {"token": "test-token"}
    assert json.loads(credentials_file.read_text(encoding="utf-8")) == credentials
    if os.name != "nt":
        assert stat.S_IMODE(credentials_file.stat().st_mode) == 0o600


@pytest.mark.parametrize("content", ['{"token": 123}', "not json"])
def test_invalid_runalyze_credentials_explain_required_format(tmp_path, content):
    sync_runalyze = load_sync_runalyze()
    credentials_file = tmp_path / "runalyze_credentials.json"
    credentials_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty 'token' string"):
        sync_runalyze["_load_runalyze_credentials"](str(credentials_file))
