import json
from pathlib import Path
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn
from runalyze import api


def load_data():
    with open("runalyze_credentials.json", "r", encoding="utf-8") as f:
        credentials = json.load(f)

    page = 1
    activities: list[api.ActivitiyType] = []

    with Progress(*Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn()) as p:
        task_id = p.add_task("Loading activities", total=None)
        while True:
            new_activities = api.get_activities(credentials, page)
            if not new_activities:
                break

            activities.extend(new_activities)
            page += 1
            p.update(task_id, advance=1)

    Path("cache").mkdir(parents=True, exist_ok=True)
    with open("cache/activities.json", "w", encoding="utf-8") as f:
        json.dump(activities, f)

    return activities


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

    Returns:
        bool: True if the activity is valid, False otherwise.
    """
    return True  # TODO: implement


def purge_fields(activity: api.ActivityType) -> None:
    """Removes fields which are not needed in place."""
    pass  # TODO: implement
