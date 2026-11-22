import datetime
import json
import os
import runpy
import stat
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from runalyze import api


def load_sync_runalyze():
    script = Path(__file__).resolve().parents[1] / "sync-runalyze.py"
    return runpy.run_path(str(script))


def runalyze_activity(activity_id, start, *, sport="Radfahren", virtual=True):
    return {
        "id": activity_id,
        "date_time": start.isoformat(),
        "sport": {"name": sport},
        "is_virtual": virtual,
    }


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


def test_get_activities_filters_at_or_after_time(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = []
    request = Mock(return_value=response)
    monkeypatch.setattr(api.requests, "get", request)
    start_time = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)

    api.get_activities({"token": "test"}, page=2, n=30, start_time=start_time)

    assert request.call_args.kwargs["params"] == {
        "page": 2,
        "order[id]": "desc",
        "itemsPerPage": 30,
        "time[gte]": int(start_time.timestamp()),
    }


def test_missing_zwift_fit_propagates_retry_error(monkeypatch, tmp_path):
    sync_runalyze = load_sync_runalyze()
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    ride = {
        "id": 7,
        "sport": "CYCLING",
        "startDate": start.isoformat(),
        "fitFileBucket": "s3-fit-zwift",
        "fitFileKey": "prod/123/7",
    }
    monkeypatch.setattr(
        sync_runalyze["zwift_sync"],
        "get_zwift_rides",
        lambda **_kwargs: [ride],
    )
    monkeypatch.setattr(sync_runalyze["zwift_sync"], "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(sync_runalyze["zwift_sync"], "download_fit", lambda *_args: False)

    with pytest.raises(RuntimeError, match="no FIT file yet"):
        sync_runalyze["sync_zwift_rides"]({1: start})


@pytest.mark.parametrize(
    "switches",
    [
        [],
        ["--sync-secondary"],
        ["--sync-zwift"],
        ["--sync-secondary", "--sync-zwift"],
    ],
)
def test_cli_selects_runalyze_activities_for_both_syncs(monkeypatch, tmp_path, switches):
    script = Path(__file__).resolve().parents[1] / "sync-runalyze.py"
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "sync-runalyze-config.txt"
    previous_checkpoint = "2025-01-01 00:00"
    checkpoint.write_text(previous_checkpoint, encoding="utf-8")
    (tmp_path / "runalyze_credentials.json").write_text('{"token": "test"}', encoding="utf-8")
    (tmp_path / "runalyze_credentials_secondary.json").write_text('{"token": "secondary"}', encoding="utf-8")

    def check_checkpoint(*_args):
        assert checkpoint.read_text(encoding="utf-8") == previous_checkpoint

    sync_zwift = Mock(side_effect=check_checkpoint)
    monkeypatch.setattr(sys, "argv", [str(script), *switches])
    module = runpy.run_path(str(script))
    monkeypatch.setitem(module["main"].__globals__, "sync_zwift_rides", sync_zwift)
    garmin_upload = Mock()
    cutoff = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    later = cutoff + datetime.timedelta(minutes=1)
    unmarked = runalyze_activity(2, later)
    del unmarked["is_virtual"]
    virtual_ride = runalyze_activity(4, cutoff)
    virtual_ride["date_time"] = "2025-01-01T00:00:00Z"
    selected = [
        runalyze_activity(1, later, virtual=None),
        unmarked,
        runalyze_activity(3, later, sport="Laufen"),
        virtual_ride,
    ]
    get_activities = Mock(side_effect=[selected, []])
    sync_trimp = Mock()
    monkeypatch.setitem(module["main"].__globals__, "upload_garmin_original_to_main", garmin_upload)
    monkeypatch.setitem(module["main"].__globals__, "get_activities", get_activities)
    monkeypatch.setitem(module["main"].__globals__, "sync_trimp_to_secondary", sync_trimp)

    module["main"]()

    garmin_upload.assert_called_once()
    assert checkpoint.read_text(encoding="utf-8") != previous_checkpoint
    if switches:
        assert get_activities.call_args_list == [
            call(
                {"token": "test"},
                page=1,
                n=30,
                start_time=cutoff,
            ),
            call(
                {"token": "test"},
                page=2,
                n=30,
                start_time=cutoff,
            ),
        ]
    else:
        get_activities.assert_not_called()
    if "--sync-secondary" in switches:
        sync_trimp.assert_called_once_with(
            [(activity, later) for activity in selected[:-1]] + [(selected[-1], cutoff)],
            {"token": "secondary"},
        )
    else:
        sync_trimp.assert_not_called()
    if "--sync-zwift" in switches:
        sync_zwift.assert_called_once_with({4: cutoff})
    else:
        sync_zwift.assert_not_called()
