import json
import requests
import pathlib

RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"

# load credentials for runalyze
with open(pathlib.Path(__file__).parent / "runalyze_credentials.json", "r", encoding="utf-8") as file:
    credentials_runalyze = json.load(file)

r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads/83430223",
                  headers=credentials_runalyze)
print(r.text)
if r.status_code != 201:
    raise Exception("Error uploading activity to Runalyze")
