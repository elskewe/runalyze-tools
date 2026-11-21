"""Sync Garmin TRIMP data to a secondary Runalyze account and optionally enrich the main account with original Garmin uploads."""
import argparse
import datetime
import json
import os
import stat
import warnings
from getpass import getpass

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError

from runalyze.api import get_activities, upload_activity
from runalyze.tcx import create, translate_activity_type

CONFIG_FILE = "sync-runalyze-config.txt"
RUNALYZE_CREDENTIALS_FILE = "runalyze_credentials.json"
SECONDARY_RUNALYZE_CREDENTIALS_FILE = "runalyze_credentials_secondary.json"


def _load_sync_start_time(
    config_file: str = CONFIG_FILE,
    now: datetime.datetime | None = None,
) -> datetime.datetime:
    try:
        with open(config_file, encoding="utf-8") as file:
            config_value = file.readline().strip()
    except FileNotFoundError:
        now = now or datetime.datetime.now(tz=datetime.UTC)
        start_time = now.astimezone(datetime.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        with open(config_file, "x", encoding="utf-8") as file:
            file.write(start_time.strftime("%Y-%m-%d %H:%M"))
        print(
            f"Created {config_file}, starting with today at 00:00 UTC. "
            "Set an earlier date and run again to import older activities."
        )
        return start_time

    try:
        return datetime.datetime.strptime(config_value, "%Y-%m-%d %H:%M").replace(
            tzinfo=datetime.UTC
        )
    except ValueError as error:
        raise ValueError(
            f"{config_file} must contain a date and time in YYYY-MM-DD HH:MM format."
        ) from error


def _load_runalyze_credentials(credentials_file: str) -> dict[str, object]:
    if not os.path.exists(credentials_file):
        token = getpass(f"Runalyze API token for {credentials_file}: ").strip()
        if not token:
            raise ValueError("Runalyze API token cannot be empty.")

        with open(
            credentials_file, "x", encoding="utf-8",
            opener=lambda path, flags: os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR),
        ) as file:
            json.dump({"token": token}, file, indent=2)
            file.write("\n")
        print(f"Created {credentials_file}.")

    try:
        with open(credentials_file, encoding="utf-8") as file:
            credentials = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{credentials_file} must contain valid JSON with a non-empty 'token' string, "
            'for example: {"token": "MYTOKEN"}.'
        ) from error
    if (
        not isinstance(credentials, dict)
        or not isinstance(credentials.get("token"), str)
        or not credentials["token"].strip()
    ):
        raise ValueError(
            f"{credentials_file} must contain a non-empty 'token' string, "
            'for example: {"token": "MYTOKEN"}.'
        )
    return credentials


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
        "--sync-secondary",
        action="store_true",
        help="Copy Garmin TRIMP values to a secondary Runalyze account as Runalyze TRIMP.",
    )
    args = parser.parse_args()

    tz_info = datetime.timezone.utc
    start_time = _load_sync_start_time()
    main_credentials = _load_runalyze_credentials(RUNALYZE_CREDENTIALS_FILE)

    upload_garmin_original_to_main(main_credentials, start_time)

    if args.sync_secondary:
        secondary_credentials = _load_runalyze_credentials(SECONDARY_RUNALYZE_CREDENTIALS_FILE)
        sync_trimp_to_secondary(main_credentials, secondary_credentials, start_time)

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(datetime.datetime.now(tz=tz_info).strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
