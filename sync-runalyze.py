"""Upload original Garmin activities and optionally sync TRIMP and match Zwift rides."""
import argparse
import datetime
import json
import os
import warnings
from collections import Counter
from getpass import getpass

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError

from runalyze.api import ActivityType, get_activities, upload_activity
from runalyze.browser import merge_activity_file
from runalyze.tcx import create, translate_activity_type
from zwift import sync as zwift_sync

CONFIG_FILE = "sync-runalyze-config.txt"
CYCLING_SPORTS = {"cycling", "biking", "virtual ride", "virtual cycling", "radfahren", "virtuelles radfahren"}
ZWIFT_START_TOLERANCE = datetime.timedelta(minutes=10)


def sync_zwift_rides(virtual_rides: dict[int, datetime.datetime]) -> None:
    if not virtual_rides:
        print("No Runalyze virtual rides since the sync cutoff to match with Zwift.")
        return

    zwift_rides = {
        int(ride["id"]): (ride, zwift_sync.ride_start_time(ride))
        for ride in zwift_sync.get_zwift_rides()
        if ride["sport"] == "CYCLING"
    }
    matches = {
        runalyze_id: [
            zwift_id for zwift_id, (_, zwift_start) in zwift_rides.items()
            if abs(runalyze_start - zwift_start) <= ZWIFT_START_TOLERANCE
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
        ride = zwift_rides[zwift_id][0]
        destination = zwift_sync.DOWNLOAD_DIR / f"runalyze-{runalyze_id}_zwift-{zwift_id}.fit"
        if not destination.is_file() and not zwift_sync.download_fit(ride, destination):
            raise RuntimeError(f"Zwift ride {zwift_id} has no FIT file yet; retry the sync later.")
        if destination.read_bytes()[8:12] != b".FIT":
            raise ValueError(f"Cached Zwift ride {zwift_id} is not a FIT file.")
        merge_activity_file(runalyze_id, destination)
        print(f"{activity_url}: merged Zwift ride {zwift_id} from {destination.resolve()}.")


def login_to_garmin(email: str | None = None, password: str | None = None) -> Garmin:
    """Initialize Garmin API with smart error handling and recovery."""
    TOKEN_STORE = os.path.expanduser("~/.garminconnect")
    # First try to login with stored tokens
    try:
        garmin = Garmin()
        garmin.login(TOKEN_STORE)
        print("Successfully logged in using stored tokens!")
        return garmin

    except (FileNotFoundError, GarminConnectAuthenticationError, GarminConnectConnectionError):
        print("No valid tokens found. Requesting fresh login credentials.")

    # Loop for credential entry with retry on auth failure
    while True:
        try:
            # Get credentials if not provided
            if not email or not password:
                email = input("Email address: ").strip()
                password = getpass("Password: ")

            print("Logging in with credentials...")
            garmin = Garmin(email=email, password=password, return_on_mfa=True)
            password = None # Don't keep the plaintext password in local scope longer than needed.
            garmin.login()

            # Save tokens for future use
            try:
                garmin.client.dump(TOKEN_STORE)
            except Exception: # If dump fails, don't prevent the login from succeeding
                print("Warning: failed to persist Garmin tokens to token store.")
            print(f"Login successful! Tokens saved to: {TOKEN_STORE}")

            return garmin

        except GarminConnectAuthenticationError as err:
            print(f"\n❌ {err}")
            print("💡 Please check your username and password and try again")
            # Clear the provided credentials to force re-entry
            email = None
            password = None
            continue


def upload_garmin_original_to_main(runalyze_credentials, start_time: datetime.datetime) -> None:
    garmin = login_to_garmin()

    start_date = start_time.date().isoformat()
    end_date = datetime.date.today().isoformat()
    activities = garmin.get_activities_by_date(start_date, end_date)
    for activity in activities:
        if "endTimeGMT" in activity:
            activity_end_time = datetime.datetime.fromisoformat(activity["endTimeGMT"]) \
                .replace(tzinfo=datetime.timezone.utc)
            if activity_end_time < start_time:
                continue  # ignore old activities to prevent superfluous uploads
        activity_id = activity["activityId"]
        original_activity = garmin.download_activity(
            str(activity_id),
            dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL,
        )
        if not original_activity:
            warnings.warn(f"Activity {activity_id} has no original Garmin data, skipping.", stacklevel=2)
            continue

        upload_activity(original_activity, runalyze_credentials, file_type="zip")


def sync_trimp_to_secondary(
    activities: list[tuple[ActivityType, datetime.datetime]], secondary_credentials
) -> None:
    for activity, activity_date in activities:
        if "fit_trimp" not in activity:
            warnings.warn(f"Activity at {activity_date} has no fit file TRIMP, skipping.")
            continue
        tcx_string = create(activity_date, activity["fit_trimp"]*60, translate_activity_type(activity["sport"]["name"]))
        upload_activity(tcx_string, secondary_credentials, title=activity["title"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Upload original Garmin activities to Runalyze, optionally copy TRIMP to a secondary "
            "account, and download and merge matching Zwift rides."
        )
    )
    parser.add_argument(
        "--sync-secondary",
        action="store_true",
        help="Copy Garmin TRIMP values to a secondary Runalyze account as Runalyze TRIMP.",
    )
    parser.add_argument(
        "--sync-zwift",
        action="store_true",
        help="Download and merge FIT files matching recent Runalyze virtual rides.",
    )
    args = parser.parse_args()

    tz_info = datetime.timezone.utc
    with open(CONFIG_FILE, encoding="utf-8") as file:
        start_time = datetime.datetime.strptime(file.readline().strip(), "%Y-%m-%d %H:%M").replace(tzinfo=tz_info)

    with open("runalyze_credentials.json", encoding="utf-8") as f:
        main_credentials = json.load(f)

    upload_garmin_original_to_main(main_credentials, start_time)

    activities = []
    if args.sync_secondary or args.sync_zwift:
        for activity in get_activities(main_credentials, n=30):
            activity_date = datetime.datetime.fromisoformat(activity["date_time"])
            if activity_date >= start_time:
                activities.append((activity, activity_date))

    if args.sync_secondary:
        with open("runalyze_credentials_secondary.json", encoding="utf-8") as f:
            secondary_credentials = json.load(f)
        sync_trimp_to_secondary(activities, secondary_credentials)

    if args.sync_zwift:
        virtual_rides = {
            activity["id"]: activity_date
            for activity, activity_date in activities
            if activity.get("is_virtual") and activity["sport"]["name"].strip().casefold() in CYCLING_SPORTS
        }
        sync_zwift_rides(virtual_rides)

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(datetime.datetime.now(tz=tz_info).strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
