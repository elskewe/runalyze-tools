import requests

RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"


def upload_activity(tcx_string: str, credentials):
    """Uploads an activity to Runalyze."""
    r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads",
                      headers=credentials,
                      files={"file": ("activity.tcx", tcx_string)})
    print(r.text)
    if r.status_code != 201:
        raise requests.exceptions.RequestException("Error uploading activity to Runalyze")
