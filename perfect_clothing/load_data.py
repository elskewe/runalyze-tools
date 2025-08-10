import json
from pathlib import Path
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn
from runalyze import api

CACHE_FILE = "cache/activities.json"


def load_data():
    with open("runalyze_credentials.json", "r", encoding="utf-8") as f:
        credentials = json.load(f)

    page = 1
    activities: list[api.ActivityType] = []

    with Progress(*Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn()) as p:
        task_id = p.add_task("Loading activities", total=None)
        while True:
            new_activities = api.get_activities(credentials, page)
            if not new_activities:
                break

            activities.extend(new_activities)
            page += 1
            p.update(task_id, advance=1)

    Path(CACHE_FILE).mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(activities, f, indent=4)

    return activities


def clean_file():
    """Calls `clean_data` on the cache file."""
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = clean_data(data)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def clean_data(data: list[api.ActivityType]) -> list[api.ActivityType]:
    """Removes activities which are not useful as well as keys which are not needed

    The former is defined in `valid_activity`, the later in `purge_fields`.
    """
    output = [e for e in data if valid_activity(e)]
    for e in output:
        purge_fields(e)
    return output


def valid_activity(activity: api.ActivityType) -> bool:
    """Checks if the activity is valid.

    This means:
        - The activity is a run
        - The activity has a temperature recorded

    Returns:
        bool: True if the activity is valid, False otherwise.
    """
    is_valid = (
        isinstance(s := activity.get("sport"), dict) and s.get("name") == "Laufen"
        and "temperature" in activity
    )
    return is_valid


def purge_fields(activity: api.ActivityType) -> None:
    """Removes fields which are not needed in place."""
    # these are checked with `startswith`
    useless_fields = [
        "aerobic_decoupling_", "max_hr_drop", "stamina_", "fit_performance_condition", "hrv",
        "groundcontact", "vertical_", "stride_length", "cadence",
        "partner"
    ]
    for k in list(activity.keys()):
        if any(k.startswith(f) for f in useless_fields):
            del activity[k]
