import requests

RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"

PossibleDictValueTypes = int | float | str | bool
# return type of the `activity` endpoint
ActivityType = dict[str, PossibleDictValueTypes | dict[str, PossibleDictValueTypes]]


def upload_activity(tcx_string: str, credentials, title: str = "", note: str = ""):
    """Uploads an activity to Runalyze."""
    r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads",
                      headers=credentials,
                      files={"file": ("activity.tcx", tcx_string)},
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


def get_activities(credentials, page=1) -> list[ActivityType]:
    """Returns the last 100 activities.

    Args:
        credentials (dict): The credentials for the Runalyze API.
        page (int, optional): The page to get. Defaults to 1.
    """
    r = requests.get(RUNALYZE_API_ENDPOINT + "activity",
                     headers=credentials,
                     params={"page": page, "order[id]": "desc"})
    if r.status_code != requests.codes.OK:
        raise requests.exceptions.RequestException(
            f"Error getting activities from Runalyze (response code: {r.status_code})"
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
