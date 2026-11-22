import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from keyring.credentials import SimpleCredential

from runalyze import browser as runalyze_browser
from zwift import sync as zwift


def load_sync_zwift_rides():
    script = Path(__file__).resolve().parents[1] / "sync-runalyze.py"
    return runpy.run_path(str(script))["sync_zwift_rides"]


def zwift_ride(ride_id, start, *, sport="CYCLING"):
    return {
        "id": ride_id,
        "sport": sport,
        "startDate": start.isoformat(),
        "fitFileBucket": "s3-fit-zwift",
        "fitFileKey": f"prod/123/{ride_id}",
    }


def test_empty_ride_selection_does_not_contact_zwift(monkeypatch):
    sync_zwift_rides = load_sync_zwift_rides()
    get_rides = Mock()
    monkeypatch.setattr(zwift, "get_zwift_rides", get_rides)

    sync_zwift_rides({})

    get_rides.assert_not_called()


def test_zwift_activity_pagination_stops_before_selected_window(monkeypatch):
    get_credential = Mock(return_value=SimpleCredential("test-user", "test-password"))
    monkeypatch.setattr(zwift.keyring, "get_credential", get_credential)
    token = Mock(status_code=200)
    token.json.return_value = {"access_token": "test-token"}
    monkeypatch.setattr(zwift.requests, "post", Mock(return_value=token))

    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first_page = [
        zwift_ride(ride_id, cutoff + timedelta(days=49 - ride_id))
        for ride_id in range(50)
    ]
    second_page = [
        zwift_ride(50, cutoff - timedelta(minutes=11)),
    ]
    profile = Mock()
    profile.json.return_value = {"id": 123}
    second = Mock()
    second.json.return_value = second_page
    second.status_code = 200
    first = Mock()
    first.json.return_value = first_page
    first.status_code = 200
    get = Mock(side_effect=[profile, first, second])
    monkeypatch.setattr(zwift.requests, "get", get)

    rides = zwift.get_zwift_rides(start_time=cutoff)

    assert [ride["id"] for ride in rides] == list(range(50))
    assert [call.kwargs["params"] for call in get.call_args_list[1:]] == [
        {"start": 0, "limit": 50},
        {"start": 50, "limit": 50},
    ]


def test_missing_saved_login_explains_setup_without_contacting_zwift(monkeypatch):
    monkeypatch.setattr(zwift.keyring, "get_credential", Mock(return_value=None))
    login = Mock()
    monkeypatch.setattr(zwift.requests, "post", login)

    with pytest.raises(RuntimeError, match="python -m keyring set runalyze-tools-zwift"):
        zwift.get_zwift_rides()

    login.assert_not_called()


def test_merge_dialog_uses_approved_date_and_field_settings():
    dialog, rows, radio, checkbox = Mock(), Mock(), Mock(), Mock()
    rows.count.return_value = 1
    radio.count.return_value = 1
    checkbox.count.return_value = 1
    rows.get_by_role.side_effect = lambda role: radio if role == "radio" else checkbox
    dialog.get_by_role.return_value = rows

    runalyze_browser._apply_merge_settings(dialog)

    radio.check.assert_called_once_with()
    assert [call.args[0] for call in checkbox.set_checked.call_args_list] == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_remerges_cached_unique_match_without_redownloading(monkeypatch, tmp_path, capsys):
    sync_zwift_rides = load_sync_zwift_rides()
    integration = sync_zwift_rides.__globals__
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
    merge = Mock()
    monkeypatch.setitem(integration, "merge_activity_file", merge)
    profile, activities, fit = Mock(), Mock(), Mock(content=b"\x0e\x10\0\0\0\0\0\0.FIT\0\0")
    profile.json.return_value = {"id": 123}
    activities.json.return_value = [
        zwift_ride(7, start + integration["ZWIFT_START_TOLERANCE"]),
        zwift_ride(8, start + integration["ZWIFT_START_TOLERANCE"] + timedelta(seconds=1)),
        zwift_ride(9, start, sport="RUNNING"),
    ]
    activities.json.return_value[0]["startDate"] = (
        start + integration["ZWIFT_START_TOLERANCE"]
    ).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    get = Mock(side_effect=[profile, activities, fit, profile, activities])
    monkeypatch.setattr(zwift.requests, "get", get)

    sync_zwift_rides(virtual_rides)
    sync_zwift_rides(virtual_rides)

    get_credential.assert_called_with("runalyze-tools-zwift", None)
    assert login.call_args.kwargs["data"]["username"] == "test-user"
    assert login.call_args.kwargs["data"]["password"] == "test-password"
    assert get.call_args_list[0].kwargs["headers"]["Accept"] == "application/json"
    assert get.call_args_list[1].kwargs["headers"]["Accept"] == "application/json"
    path = tmp_path / "runalyze-1_zwift-7.fit"
    assert list(tmp_path.glob("*.fit")) == [path]
    assert path.read_bytes() == fit.content
    assert [call.args for call in merge.call_args_list] == [(1, path), (1, path)]
    assert get.call_args_list[1].args[0].endswith("/profiles/123/activities/")
    assert get.call_args_list[2].args == ("https://s3-fit-zwift.s3.amazonaws.com/prod/123/7",)
    assert "headers" not in get.call_args_list[2].kwargs
    output = capsys.readouterr().out
    assert output.count(f"https://runalyze.com/activity/1: merged Zwift ride 7 from {path}") == 2


@pytest.mark.parametrize(("runalyze_count", "zwift_count"), [(1, 2), (2, 1)])
def test_ambiguous_match_is_not_downloaded(monkeypatch, runalyze_count, zwift_count):
    sync_zwift_rides = load_sync_zwift_rides()
    start = datetime.now(timezone.utc)
    virtual_rides = dict.fromkeys(range(runalyze_count), start)
    monkeypatch.setattr(zwift, "get_zwift_rides", Mock(return_value=[
        zwift_ride(ride_id, start) for ride_id in range(zwift_count)
    ]))
    download = Mock()
    monkeypatch.setattr(zwift, "download_fit", download)
    merge = Mock()
    monkeypatch.setitem(sync_zwift_rides.__globals__, "merge_activity_file", merge)

    with pytest.warns(UserWarning, match="Ambiguous Zwift match"):
        sync_zwift_rides(virtual_rides)

    download.assert_not_called()
    merge.assert_not_called()


