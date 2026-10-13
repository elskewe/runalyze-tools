import re
from datetime import datetime

import requests

RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"

PossibleDictValueTypes = int | float | str | bool
# return type of the `activity` endpoint
ActivityType = dict[str, PossibleDictValueTypes | dict[str, PossibleDictValueTypes]]


def _extract_tcx_start_time(tcx_string: str) -> str:
    match = re.search(r"<Id>([^<]+)</Id>", tcx_string)
    if not match:
        return ""

    timestamp = match.group(1).strip()
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d_%H-%M")


def upload_activity(activity_data: str | bytes, credentials, title: str = "", note: str = "", file_type: str = "tcx"):
    """Uploads an activity to Runalyze.

    Args:
        activity_data: The raw activity payload, either as a TCX XML string, raw FIT bytes, or an original Garmin ZIP payload.
        credentials: Runalyze API headers.
        title: Optional Runalyze title.
        note: Optional Runalyze note.
        file_type: The file extension to use for the upload. Defaults to "tcx".
    """
    file_type = file_type.lower().lstrip(".")
    if file_type == "tcx":
        if not isinstance(activity_data, str):
            raise TypeError("TCX uploads require a string payload.")
        file_name = f"{_extract_tcx_start_time(activity_data)}.tcx"
    elif file_type in {"fit", "zip"}:
        if not isinstance(activity_data, bytes):
            raise TypeError("FIT/ZIP uploads require a bytes payload.")
        file_name = f"activity.{file_type}"
    else:
        raise ValueError(f"Unsupported file_type '{file_type}'. Supported types: tcx, fit, zip")

    r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads",
                      headers=credentials,
                      files={"file": (file_name, activity_data)},
                      data={"title": title, "note": note})
    print(r.text)
    if r.status_code != requests.codes.CREATED:
        raise requests.exceptions.RequestException("Error uploading activity to Runalyze")


def upload_body_composition(data: dict, credentials):
    """Uploads body composition data to Runalyze."""
    r = requests.post(RUNALYZE_API_ENDPOINT + "metrics/bodyComposition",
                      headers=credentials,
                      json=data)
    print(r.text)
    if r.status_code != requests.codes.CREATED:
        raise requests.exceptions.RequestException("Error uploading body composition data to Runalyze")


def get_activities(credentials, page=1, n=100) -> list[ActivityType]:
    """Returns the last n activities.

    Args:
        credentials (dict): The credentials for the Runalyze API.
        page (int, optional): The page to get. Defaults to 1.
        n (int, optional): The number of activities to get. Defaults to 100. Note that big values (like 500) might lead to a 524 response code
    """
    r = requests.get(RUNALYZE_API_ENDPOINT + "activity",
                     headers=credentials,
                     params={"page": page, "order[id]": "desc", "itemsPerPage": n})
    if r.status_code != requests.codes.OK:
        raise requests.exceptions.RequestException(
            f"Error getting activities from Runalyze (response code: {r.status_code} {r.reason})"
        )
    return r.json()


def get_body_composition(credentials, page=1) -> list[ActivityType]:
    """Returns the last 100 body composition data points.

    Note that this always return the oldest data points first.

    Args:
        credentials (dict): The credentials for the Runalyze API.
        page (int, optional): The page to get. Defaults to 1.
    """
    r = requests.get(RUNALYZE_API_ENDPOINT + "metrics/bodyComposition",
                     headers=credentials, params={"page": page})
    if r.status_code != requests.codes.OK:
        raise requests.exceptions.RequestException(
            f"Error getting body composition data from Runalyze (response code: {r.status_code})"
        )
    return r.json()
