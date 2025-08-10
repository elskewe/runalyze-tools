import json
from pathlib import Path
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn
from runalyze import api

def load_data():
    with open("runalyze_credentials.json", "r", encoding="utf-8") as f:
        credentials = json.load(f)

    page = 1
    activities = []

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

