"""Syncs the Garmin TRIMP from the Runalyze activities to the secondary Runalyze account."""
import json
import datetime

from runalyze.api import get_activities, upload_activity
from runalyze.tcx import create, translate_activity_type

CONFIG_FILE = "sync-runalyze-config.txt"


def main():
    with open("runalyze_credentials.json", "r", encoding="utf-8") as f:
        main_credentials = json.load(f)

    with open("runalyze_credentials_secondary.json", "r", encoding="utf-8") as f:
        secondary_credentials = json.load(f)

    tz_info = datetime.timezone(datetime.timedelta(hours=1), "Europe/Berlin")
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        start_time = datetime.datetime.strptime(file.readline().strip(), "%Y-%m-%d %H:%M").replace(tzinfo=tz_info)

    activities = get_activities(main_credentials)
    for activity in activities:
        if (activity_date := datetime.datetime.fromisoformat(activity["date_time"])) < start_time:
            continue  # ignore old activities to prevent duplicates
        tcx_string = create(activity_date, activity["fit_trimp"]*60, translate_activity_type(activity["sport"]["name"]))
        upload_activity(tcx_string, secondary_credentials, title=activity["title"])

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(datetime.datetime.now(tz=tz_info).strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
