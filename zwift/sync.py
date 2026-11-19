"""Match Zwift rides to Runalyze activities and download FIT files for manual merging."""

import re
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import keyring
import requests
from keyring.errors import KeyringError

ZWIFT_API = "https://us-or-rly101.zwift.com/api"
ZWIFT_TOKEN_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/token"
ZWIFT_KEYRING_SERVICE = "runalyze-tools-zwift"
START_TOLERANCE = timedelta(minutes=10)
DOWNLOAD_DIR = Path("cache", "zwift")
TIMEOUT = 30


class ZwiftRide(TypedDict):
    id: int
    sport: str
    startDate: str
    fitFileBucket: str | None
    fitFileKey: str | None


def _utc_datetime(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"Expected a timezone offset on activity timestamp {value!r}")
    return timestamp.astimezone(timezone.utc)


def _get_zwift_rides() -> list[ZwiftRide]:
    try:
        credential = keyring.get_credential(ZWIFT_KEYRING_SERVICE, None)
    except KeyringError as exc:
        raise RuntimeError(f"Could not read the Zwift login from the system credential store: {exc}") from exc
    if credential is None or not credential.username or not credential.password:
        raise RuntimeError(
            "No saved Zwift login. Run "
            f"python -m keyring set {ZWIFT_KEYRING_SERVICE} YOUR_ZWIFT_EMAIL"
        )

    response = requests.post(
        ZWIFT_TOKEN_URL,
        data={
            "client_id": "Zwift Game Client",
            "grant_type": "password",
            "username": credential.username,
            "password": credential.password,
        },
        timeout=TIMEOUT,
        allow_redirects=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Zwift login failed (HTTP {response.status_code}).")
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}

    response = requests.get(f"{ZWIFT_API}/profiles/me", headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    profile_id = int(response.json()["id"])
    response = requests.get(
        f"{ZWIFT_API}/profiles/{profile_id}/activities/",
        headers=headers,
        params={"start": 0, "limit": 50},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    rides = response.json()
    if not isinstance(rides, list):
        raise ValueError("Zwift did not return an activity list.")
    return rides


def _download_fit(ride: ZwiftRide, destination: Path) -> bool:
    bucket, key = ride.get("fitFileBucket"), ride.get("fitFileKey")
    if not bucket or not key:
        warnings.warn(f"Zwift ride {ride['id']} has no FIT file yet.", stacklevel=2)
        return False
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
        raise ValueError(f"Invalid FIT bucket for Zwift ride {ride['id']}.")

    response = requests.get(f"https://{bucket}.s3.amazonaws.com/{quote(key)}", timeout=TIMEOUT)
    response.raise_for_status()
    if response.content[8:12] != b".FIT":
        raise ValueError(f"Zwift ride {ride['id']} did not return a FIT file.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.replace(destination)
    return True


def sync_zwift_rides(virtual_rides: dict[int, datetime]) -> None:
    if not virtual_rides:
        print("No Runalyze virtual rides since the sync cutoff to match with Zwift.")
        return

    zwift_rides = {
        int(ride["id"]): (ride, _utc_datetime(ride["startDate"]))
        for ride in _get_zwift_rides()
        if ride["sport"] == "CYCLING"
    }
    matches = {
        runalyze_id: [
            zwift_id for zwift_id, (_, zwift_start) in zwift_rides.items()
            if abs(runalyze_start - zwift_start) <= START_TOLERANCE
        ]
        for runalyze_id, runalyze_start in virtual_rides.items()
    }
    match_counts = Counter(zwift_id for candidates in matches.values() for zwift_id in candidates)
    for runalyze_id, candidates in matches.items():
        activity_url = f"https://runalyze.com/activity/{runalyze_id}"
        if not candidates:
            print(f"No matching Zwift ride for {activity_url}.")
            continue
        if len(candidates) != 1 or match_counts[candidates[0]] != 1:
            warnings.warn(f"Ambiguous Zwift match for {activity_url}; skipping.", stacklevel=2)
            continue
        zwift_id = candidates[0]
        destination = DOWNLOAD_DIR / f"runalyze-{runalyze_id}_zwift-{zwift_id}.fit"
        if destination.is_file() or _download_fit(zwift_rides[zwift_id][0], destination):
            print(f"{activity_url}: use 'Merge file into activity' with {destination.resolve()}")
