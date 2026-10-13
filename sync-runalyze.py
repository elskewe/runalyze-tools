"""Sync Garmin TRIMP data to a secondary Runalyze account and optionally enrich the main account with original Garmin uploads."""
import argparse
import datetime
import json
import os
import warnings
from getpass import getpass

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError

from runalyze.api import get_activities, upload_activity
from runalyze.tcx import create, translate_activity_type

CONFIG_FILE = "sync-runalyze-config.txt"


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
            garmin.client.dump(TOKEN_STORE)
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
        activity_id = activity["activityId"]
        original_activity = garmin.download_activity(
            str(activity_id),
            dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL,
        )
        if not original_activity:
            warnings.warn(f"Activity {activity_id} has no original Garmin data, skipping.", stacklevel=2)
            continue

        upload_activity(original_activity, runalyze_credentials, file_type="zip")


def sync_trimp_to_secondary(main_credentials, secondary_credentials, start_time: datetime.datetime) -> None:
    # as syncing is rather frequent its sufficient to get only a limited number of activities to
    # speed up the download
    activities = get_activities(main_credentials, n=30)
    for activity in activities:
        if (activity_date := datetime.datetime.fromisoformat(activity["date_time"])) < start_time:
            continue  # ignore old activities to prevent duplicates
        if "fit_trimp" not in activity:
            warnings.warn(f"Activity at {datetime.datetime.fromisoformat(activity['date_time']) } has no fit file TRIMP, skipping.")
            continue
        tcx_string = create(activity_date, activity["fit_trimp"]*60, translate_activity_type(activity["sport"]["name"]))
        upload_activity(tcx_string, secondary_credentials, title=activity["title"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sync Garmin TRIMP values to a secondary Runalyze account and optionally enrich the "
            "main account with original Garmin uploads."
        )
    )
    parser.add_argument(
        "--upload-original",
        action="store_true",
        help="Upload recent original Garmin fit files to the main Runalyze account.",
    )
    args = parser.parse_args()

    tz_info = datetime.timezone(datetime.timedelta(hours=1), "Europe/Berlin")
    with open(CONFIG_FILE, encoding="utf-8") as file:
        start_time = datetime.datetime.strptime(file.readline().strip(), "%Y-%m-%d %H:%M").replace(tzinfo=tz_info)

    with open("runalyze_credentials.json", encoding="utf-8") as f:
        main_credentials = json.load(f)

    if args.upload_original:
        upload_garmin_original_to_main(main_credentials, start_time)

    with open("runalyze_credentials_secondary.json", encoding="utf-8") as f:
        secondary_credentials = json.load(f)

    sync_trimp_to_secondary(main_credentials, secondary_credentials, start_time)

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(datetime.datetime.now(tz=tz_info).strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
