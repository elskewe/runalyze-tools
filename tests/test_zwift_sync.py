import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from keyring.credentials import SimpleCredential

from zwift import sync as zwift


def runalyze_ride(ride_id, start, *, sport="Radfahren", virtual=True):
    return {
        "id": ride_id,
        "date_time": start.isoformat(),
        "sport": {"name": sport},
        "is_virtual": virtual,
    }


def zwift_ride(ride_id, start, *, sport="CYCLING"):
    return {
        "id": ride_id,
        "sport": sport,
        "startDate": start.isoformat(),
        "fitFileBucket": "s3-fit-zwift",
        "fitFileKey": f"prod/123/{ride_id}",
    }


def test_empty_ride_selection_does_not_contact_zwift(monkeypatch):
    login = Mock()
    get_credential = Mock()
    monkeypatch.setattr(zwift.keyring, "get_credential", get_credential)
    monkeypatch.setattr(zwift.requests, "post", login)

    zwift.sync_zwift_rides({})

    get_credential.assert_not_called()
    login.assert_not_called()


def test_missing_saved_login_explains_setup_without_contacting_zwift(monkeypatch):
    monkeypatch.setattr(zwift.keyring, "get_credential", Mock(return_value=None))
    login = Mock()
    monkeypatch.setattr(zwift.requests, "post", login)

    with pytest.raises(RuntimeError, match="python -m keyring set runalyze-tools-zwift"):
        zwift.sync_zwift_rides({1: datetime.now(timezone.utc)})

    login.assert_not_called()


def test_downloads_unique_match_and_prints_manual_merge_link(monkeypatch, tmp_path, capsys):
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    local_start = start.astimezone(timezone(timedelta(hours=1)))
    virtual_rides = {1: local_start}
    monkeypatch.setattr(zwift, "DOWNLOAD_DIR", tmp_path)
    get_credential = Mock(return_value=SimpleCredential("test-user", "test-password"))
    monkeypatch.setattr(zwift.keyring, "get_credential", get_credential)
    token = Mock(status_code=200)
    token.json.return_value = {"access_token": "test-token"}
    login = Mock(return_value=token)
    monkeypatch.setattr(zwift.requests, "post", login)
    profile, activities, fit = Mock(), Mock(), Mock(content=b"\x0e\x10\0\0\0\0\0\0.FIT\0\0")
    profile.json.return_value = {"id": 123}
    activities.json.return_value = [
        zwift_ride(7, start + zwift.START_TOLERANCE),
        zwift_ride(8, start + zwift.START_TOLERANCE + timedelta(seconds=1)),
        zwift_ride(9, start, sport="RUNNING"),
    ]
    activities.json.return_value[0]["startDate"] = (start + zwift.START_TOLERANCE).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    get = Mock(side_effect=[profile, activities, fit, profile, activities])
    monkeypatch.setattr(zwift.requests, "get", get)

    zwift.sync_zwift_rides(virtual_rides)
    zwift.sync_zwift_rides(virtual_rides)

    get_credential.assert_called_with("runalyze-tools-zwift", None)
    assert login.call_args.kwargs["data"]["username"] == "test-user"
    assert login.call_args.kwargs["data"]["password"] == "test-password"
    assert get.call_args_list[0].kwargs["headers"]["Accept"] == "application/json"
    assert get.call_args_list[1].kwargs["headers"]["Accept"] == "application/json"
    path = tmp_path / "runalyze-1_zwift-7.fit"
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == fit.content
    assert get.call_args_list[1].args[0].endswith("/profiles/123/activities/")
    assert get.call_args_list[2].args == ("https://s3-fit-zwift.s3.amazonaws.com/prod/123/7",)
    assert "headers" not in get.call_args_list[2].kwargs
    assert capsys.readouterr().out.count(f"https://runalyze.com/activity/1: use 'Merge file into activity' with {path}") == 2


@pytest.mark.parametrize(("runalyze_count", "zwift_count"), [(1, 2), (2, 1)])
def test_ambiguous_match_is_not_downloaded(monkeypatch, runalyze_count, zwift_count):
    start = datetime.now(timezone.utc)
    virtual_rides = dict.fromkeys(range(runalyze_count), start)
    monkeypatch.setattr(zwift, "_get_zwift_rides", Mock(return_value=[
        zwift_ride(ride_id, start) for ride_id in range(zwift_count)
    ]))
    download = Mock()
    monkeypatch.setattr(zwift, "_download_fit", download)

    with pytest.warns(UserWarning, match="Ambiguous Zwift match"):
        zwift.sync_zwift_rides(virtual_rides)

    download.assert_not_called()


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
    monkeypatch.setattr(zwift, "sync_zwift_rides", sync_zwift)
    monkeypatch.setattr(sys, "argv", [str(script), *switches])
    module = runpy.run_path(str(script))
    garmin_upload = Mock()
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    later = cutoff + timedelta(minutes=1)
    unmarked = runalyze_ride(2, later)
    del unmarked["is_virtual"]
    virtual_ride = runalyze_ride(4, cutoff)
    virtual_ride["date_time"] = "2025-01-01T00:00:00Z"
    selected = [
        runalyze_ride(1, later, virtual=None),
        unmarked,
        runalyze_ride(3, later, sport="Laufen"),
        virtual_ride,
    ]
    get_activities = Mock(return_value=[*selected, runalyze_ride(5, cutoff - timedelta(seconds=1))])
    sync_trimp = Mock()
    monkeypatch.setitem(module["main"].__globals__, "upload_garmin_original_to_main", garmin_upload)
    monkeypatch.setitem(module["main"].__globals__, "get_activities", get_activities)
    monkeypatch.setitem(module["main"].__globals__, "sync_trimp_to_secondary", sync_trimp)

    module["main"]()

    garmin_upload.assert_called_once()
    assert checkpoint.read_text(encoding="utf-8") != previous_checkpoint
    if switches:
        get_activities.assert_called_once_with({"token": "test"}, n=30)
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
