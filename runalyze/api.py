import requests

RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"


def upload_activity(tcx_string: str, credentials, title: str = "", note: str = ""):
    """Uploads an activity to Runalyze."""
    r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads",
                      headers=credentials,
                      files={"file": ("activity.tcx", tcx_string)},
                      data={"title": title, "note": note})
    print(r.text)
    if r.status_code != 201:
        raise requests.exceptions.RequestException("Error uploading activity to Runalyze")


def get_activities(credentials, page=1):
def get_activities(credentials):
    """Returns the last 100 activities."""
    r = requests.get(RUNALYZE_API_ENDPOINT + "activity",
                     headers=credentials,
                     params={"page": page, "order[id]": "desc"})
    return r.json()
