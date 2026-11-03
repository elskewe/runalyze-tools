import datetime
import json
import os
import warnings
import webbrowser
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import googleapiclient.discovery
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import humanize

from runalyze.api import upload_activity
from runalyze.tcx import create_from_distance_and_pace

SCOPES = ["https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly"]
GOOGLE_CREDENTIALS_CACHE_FILE = "google_credentials.json"
GOOGLE_CLIENT_SECRETS_FILE = "google_client_secret.json"
GOOGLE_REDIRECT_URI = "http://localhost:8080/"
GOOGLE_HEALTH_SERVICE_NAME = "health"
GOOGLE_HEALTH_API_VERSION = "v4"
GOOGLE_HEALTH_DISTANCE_PARENT = "users/me/dataTypes/distance"
GOOGLE_HEALTH_EXERCISE_PARENT = "users/me/dataTypes/exercise"
RUNALYZE_CREDENTIALS_FILE = "runalyze_credentials.json"
CONFIG_FILE = "google-fit-to-runalyze-config.txt"
DEFAULT_TIME_ZONE = ZoneInfo("Europe/Berlin")
DAILY_ROLLUP_MAX_RANGE_DAYS = 90
DAILY_ROLLUP_PAGE_SIZE = 90
EXERCISE_PAGE_SIZE = 25
MILLIMETERS_PER_METER = 1000
METERS_PER_KILOMETER = 1000
MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
MAX_DISTANCE_WITHOUT_CONFIRMATION = 8000  # maximum walking distance which is synced without user confirmation
# minimum walking distance for which the user is asked for confirmation (below this the distance is so negligible that
# asking the user is not worth it)
MIN_DISTANCE_FOR_CONFIRMATION = 500
FASTEST_NORMAL_WALKING_PACE = 7.0  # in minutes per kilometer

# https://developers.google.com/health/reference/rest/v4/users.dataTypes.dataPoints#Exercise.ExerciseType
BIKING_ACTIVITY_TYPES = {"BIKING"}
WALKING_ACTIVITY_TYPES = {"WALKING", "RUNNING"}


def authorize_with_google(scopes: list[str], credentials_cache_file: str):
    """Load, refresh, or obtain Google OAuth credentials and cache them."""
    credentials = None
    if os.path.exists(credentials_cache_file):
        credentials = Credentials.from_authorized_user_file(credentials_cache_file, scopes)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError:
                credentials = None
                warnings.warn("Failed to refresh credentials", stacklevel=2)
        if not credentials:
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CLIENT_SECRETS_FILE,
                scopes=scopes,
                redirect_uri=GOOGLE_REDIRECT_URI,
            )
            authorization_url, _ = flow.authorization_url(
                access_type="offline",
                prompt="consent",
            )
            print("Opening Google Health authorization in your browser...")
            webbrowser.open(authorization_url)
            credentials = flow.run_local_server(port=8080)

        with open(credentials_cache_file, "w", encoding="utf-8") as file:
            file.write(credentials.to_json())

    return credentials


def _request_daily_rollup(health_service, start_date: datetime.date,
                          end_date: datetime.date) -> list[dict]:
    """Fetch daily distance rollups, splitting long ranges and following pages."""
    rollup_data = []
    range_start = start_date
    while range_start < end_date:
        range_end = min(range_start + datetime.timedelta(days=DAILY_ROLLUP_MAX_RANGE_DAYS), end_date)
        request_body = {
            "range": {
                "start": {"date": {"year": range_start.year, "month": range_start.month,
                                    "day": range_start.day}},
                "end": {"date": {"year": range_end.year, "month": range_end.month,
                                  "day": range_end.day}},
            },
            "windowSizeDays": 1,
            "pageSize": DAILY_ROLLUP_PAGE_SIZE,
        }
        page_token = None
        while True:
            if page_token:
                request_body["pageToken"] = page_token
            page = health_service.users().dataTypes().dataPoints().dailyRollUp(
                parent=GOOGLE_HEALTH_DISTANCE_PARENT,
                body=request_body,
            ).execute()
            rollup_data.extend(page.get("rollupDataPoints", []))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        range_start = range_end
    return rollup_data


def get_exercise_data(health_service, start_date: datetime.date,
                      end_date: datetime.date) -> list[dict]:
    """Fetch exercise sessions whose civil start dates fall in the requested range."""
    exercise_data = []
    page_token = None
    filters = (
        f'exercise.interval.civil_start_time >= "{start_date.isoformat()}" AND '
        f'exercise.interval.civil_start_time < "{end_date.isoformat()}"'
    )
    while True:
        page = health_service.users().dataTypes().dataPoints().list(
            parent=GOOGLE_HEALTH_EXERCISE_PARENT,
            filter=filters,
            pageSize=EXERCISE_PAGE_SIZE,
            pageToken=page_token,
        ).execute()
        exercise_data.extend(page.get("dataPoints", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            return exercise_data


def _date_from_health_time(civil_time: dict) -> datetime.date:
    """Convert a Google Health civil date object to a Python date."""
    date_value = civil_time["date"]
    return datetime.date(date_value["year"], date_value["month"], date_value["day"])


def extract_distance_data(rollup_data: list[dict]) -> dict[datetime.date, int]:
    """Map daily rollups to dates and convert distance totals to whole meters."""
    distances = {}
    for rollup in rollup_data:
        distance = rollup.get("distance")
        if not distance:
            continue
        date = _date_from_health_time(rollup["civilStartTime"])
        distances[date] = int(distance["millimetersSum"]) // MILLIMETERS_PER_METER
    return distances


def _exercise_date_from_interval(interval: dict) -> datetime.date:
    """Return the local calendar date for a Google Health exercise interval."""
    civil_start_time = interval.get("civilStartTime")
    if civil_start_time:
        return _date_from_health_time(civil_start_time)

    start_time = datetime.datetime.fromisoformat(interval["startTime"].replace("Z", "+00:00"))
    return start_time.astimezone(DEFAULT_TIME_ZONE).date()


def _exercise_duration_ms(exercise: dict) -> int:
    """Convert a Google Health exercise duration to milliseconds."""
    active_duration = exercise.get("activeDuration")
    if active_duration:
        return int(float(active_duration.removesuffix("s")) * MILLISECONDS_PER_SECOND)

    interval = exercise["interval"]
    start_time = datetime.datetime.fromisoformat(interval["startTime"].replace("Z", "+00:00"))
    end_time = datetime.datetime.fromisoformat(interval["endTime"].replace("Z", "+00:00"))
    return int((end_time - start_time).total_seconds() * MILLISECONDS_PER_SECOND)


def _exercise_distance_meters(exercise: dict) -> int | None:
    """Return the exercise distance in meters if Google provided it."""
    distance = exercise.get("distance")
    if not distance:
        return None

    if isinstance(distance, dict):
        distance_value = distance.get("millimeters")
        if distance_value is None:
            distance_value = distance.get("millimetersSum")
    else:
        distance_value = distance

    if distance_value is None:
        return None

    return int(distance_value) // MILLIMETERS_PER_METER


def _summarize_exercise_data(data_points: list[dict], activity_types: set[str], *,
                             include_distance: bool = False) -> dict[datetime.date, dict[str, int]]:
    """Aggregate exercise entries by day, optionally including distance totals."""
    summaries = {}
    for data_point in data_points:
        exercise = data_point["exercise"]
        if exercise["exerciseType"] not in activity_types:
            continue

        date = _exercise_date_from_interval(exercise["interval"])
        summary = summaries.setdefault(date, {"duration": 0, "num_segments": 0, "distance": 0})
        summary["duration"] += _exercise_duration_ms(exercise)
        summary["num_segments"] += 1
        if include_distance:
            distance_meters = _exercise_distance_meters(exercise)
            if distance_meters is not None:
                summary["distance"] += distance_meters
    return summaries


def extract_activity_data(data_points: list[dict], activity_types: set[str]) -> dict[datetime.date, dict[str, int]]:
    """Sum matching exercise durations and counts by the default timezone date."""
    activities = _summarize_exercise_data(data_points, activity_types)
    return {
        date: {
            "duration": summary["duration"],
            "num_segments": summary["num_segments"],
        }
        for date, summary in activities.items()
    }


def extract_bike_distance_data(data_points: list[dict]) -> dict[datetime.date, dict[str, int]]:
    """Sum bike distances and their corresponding bike duration for entries that include a distance value."""
    distances = _summarize_exercise_data(data_points, BIKING_ACTIVITY_TYPES, include_distance=True)
    return {
        date: {
            "distance": summary["distance"],
            "duration": summary["duration"],
        }
        for date, summary in distances.items()
        if summary["distance"] > 0
    }


def _activity_datetime_for_date(date: datetime.date) -> datetime.datetime:
    """Create a noon-UTC TCX timestamp to keep the intended calendar date stable."""
    return datetime.datetime.combine(
        date,
        datetime.time(hour=12),
        tzinfo=datetime.timezone.utc,
    )


def get_user_confirmation(base_prompt: str, edit_prompt: str) -> tuple[str, str]:
    """Prompt until the user confirms a value, skips it, or enters an edit."""
    answer = ""
    while True:
        # ask user to confirm the distance, give the correct distance or skip the day
        mode = input(base_prompt + "yes/skip/edit: ")
        if mode == "yes" or mode == "y":
            mode = "yes"
            break
        elif mode == "skip" or mode == "s":
            mode = "skip"
            break
        elif mode == "edit" or mode == "e":
            mode = "edit"
            answer = input(edit_prompt)
            break
        else:
            print("Please enter yes, skip or edit")
            continue
    return (answer, mode)


def main():
    """Sync daily Google Health walking distance to Runalyze with anomaly checks."""
    credentials = authorize_with_google(SCOPES, GOOGLE_CREDENTIALS_CACHE_FILE)
    print("Successfully authorized with Google")
    health_service = googleapiclient.discovery.build(
        GOOGLE_HEALTH_SERVICE_NAME,
        GOOGLE_HEALTH_API_VERSION,
        credentials=credentials,
    )

    # load credentials for runalyze
    with open(RUNALYZE_CREDENTIALS_FILE, "r", encoding="utf-8") as file:
        credentials_runalyze = json.load(file)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        start_date = datetime.date.fromisoformat(file.readline().strip())
    end_date = datetime.datetime.now(DEFAULT_TIME_ZONE).date()

    distance_data = extract_distance_data(
        _request_daily_rollup(health_service, start_date, end_date)
    )
    exercise_data = get_exercise_data(health_service, start_date, end_date)
    biking_data = extract_activity_data(exercise_data, BIKING_ACTIVITY_TYPES)
    bike_distance_data = extract_bike_distance_data(exercise_data)
    walking_data = extract_activity_data(exercise_data, WALKING_ACTIVITY_TYPES)

    def write_config(date: datetime.date):
        """Advance the sync cursor to the day after the processed date."""
        next_date = date + datetime.timedelta(days=1)
        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            file.write(next_date.strftime("%Y-%m-%d"))

    for date in sorted(distance_data):
        bike_distance_entry = bike_distance_data.get(date, {"distance": 0, "duration": 0})
        bike_distance = bike_distance_entry["distance"]
        distance = distance_data[date] - bike_distance

        confirmation = None
        if date in biking_data:
            confirmation = (
                f"On {date.strftime('%Y-%m-%d')} you biked {biking_data[date]['num_segments']} times for a "
                f"total of {humanize.precisedelta(datetime.timedelta(milliseconds=biking_data[date]['duration']))}. "
                f"Google provided a distance for "
                f"{humanize.precisedelta(datetime.timedelta(milliseconds=bike_distance_entry['duration']))}"
                f" of that biking time, which corresponds to {bike_distance / METERS_PER_KILOMETER} km. "
                f"Is the remaining walking distance of {distance / METERS_PER_KILOMETER} km correct nevertheless? "
            )
        elif distance > MAX_DISTANCE_WITHOUT_CONFIRMATION:
            confirmation = (
                f"On {date.strftime('%Y-%m-%d')} you walked {distance/METERS_PER_KILOMETER} km. Is this correct? "
            )
        elif distance == 0:
            continue  # no distance to sync, so skip this day
        elif date in walking_data:
            pace = (
                walking_data[date]["duration"]
                / MILLISECONDS_PER_SECOND
                / SECONDS_PER_MINUTE
            ) / (distance / METERS_PER_KILOMETER)
            if pace < FASTEST_NORMAL_WALKING_PACE:
                confirmation = f"On {date.strftime('%Y-%m-%d')} you walked {distance/METERS_PER_KILOMETER} km in " + \
                               humanize.precisedelta(datetime.timedelta(milliseconds=walking_data[date]['duration']),
                                                     format="%.1f", minimum_unit="minutes") + \
                               f" at {pace:.1f} min/km. Is this correct? "

        # if there was an irregularity, let the user confirm the distance, but only if it is above a certain threshold
        if confirmation and distance > MIN_DISTANCE_FOR_CONFIRMATION:
            answer, mode = get_user_confirmation(confirmation, "Please enter the correct distance in km: ")
            if mode == "skip":
                write_config(date)
                continue
            elif mode == "edit":
                distance = int(float(answer) * METERS_PER_KILOMETER)

        activity_date = _activity_datetime_for_date(date)
        tcx_string = create_from_distance_and_pace(activity_date, distance)
        upload_activity(tcx_string, credentials_runalyze)
        write_config(date)

    print("Done")


if __name__ == "__main__":
    main()
