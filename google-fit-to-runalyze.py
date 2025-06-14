import datetime
import json
import warnings
import os

import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import googleapiclient.discovery
import humanize

SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read",
          "https://www.googleapis.com/auth/fitness.location.read"]
GOOGLE_CREDENTIALS_CACHE_FILE = "google_credentials.json"
RUNALYZE_CREDENTIALS_FILE = "runalyze_credentials.json"
CONFIG_FILE = "config.txt"
MAX_DISTANCE_WITHOUT_CONFIRMATION = 8000  # maximum walking distance which is synced without user confirmation
# minimum walking distance for which the user is asked for confirmation (below this the distance is so negligible that
# asking the user is not worth it)
MIN_DISTANCE_FOR_CONFIRMATION = 500
FASTEST_NORMAL_WALKING_PACE = 7.0  # in minutes per kilometer

# https://developers.google.com/fit/rest/v1/reference/activity-types
BIKING_ACTIVITY_IDS = [1]
WALKING_ACTIVITY_IDS = [7, 8]  # running and walking


def authorize_with_google(scopes: list[str], credentials_cache_file: str):
    credentials = None
    # Check if credentials are already cached
    if os.path.exists(credentials_cache_file):
        credentials = Credentials.from_authorized_user_file(credentials_cache_file, scopes)

    # if credentials are not cached or expired, ask for user to log in
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except:
                credentials = None
                warnings.warn("Failed to refresh credentials")
        if not credentials:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", scopes)
            credentials = flow.run_local_server(port=8081)

        # save credentials to file
        with open(credentials_cache_file, "w", encoding="utf-8") as file:
            file.write(credentials.to_json())

    return credentials


def get_aggregated_data_from_google_fit(credentials, data_source_id, start_time, end_time):
    fitness_service = googleapiclient.discovery.build('fitness', 'v1', credentials=credentials)

    aggregated_data = fitness_service.users().dataset().aggregate(
       userId='me',
       body={
           'aggregateBy': [{
              "dataSourceId": data_source_id
           }],
           'bucketByTime': {"period": {
              "type": "day",
              "value": 1,
              "timeZoneId": "Europe/Berlin"}
           },
           'startTimeMillis': start_time.timestamp() * 1000,
           'endTimeMillis': end_time.timestamp() * 1000
       }
    ).execute()

    return aggregated_data


def extract_data_from_aggregated_data(aggregated_data: dict) -> dict[datetime.datetime, list[list[int | float]]]:
    data = {}
    for bucket in aggregated_data['bucket']:
        dataset = bucket['dataset']
        if len(dataset) > 1:
            raise ValueError("More than one point in dataset")
        # get date by using the average of the start and end time which should be the right date for
        # timezones around utc
        date = datetime.datetime.fromtimestamp(
            (int(bucket['startTimeMillis']) + int(bucket['endTimeMillis'])) / 2 / 1000)
        if date in data:
            raise ValueError("Duplicate date in dataset")
        point = dataset[0]['point']
        if len(point) == 0:
            continue
        # convert list of dicts within a dict within a list into a list of lists with either the
        # `fpVal` or `intVal` values
        data[date] = [[e.get("fpVal", e.get("intVal")) for e in p["value"]] for p in point]
    return data


def extract_acitivity_segment_data(data: dict[datetime.datetime, list[list[int | float]]],
                                   activity_ids: list[int]) -> dict[datetime.datetime, dict[str, int | float]]:
    """"Takes the output of `extract_data_from_aggregated_data` and returns a list of dicts which
    only contains the data for the selected activity ID in readable form"""
    # doing this with a single list comprehension would probably come at the cost of a much worse
    # time complexity (c.f. https://stackoverflow.com/a/62315234)
    tuple_data = [(date, {"duration": l[1], "num_segments": l[2]}) for date, v in data.items() for l in v if l[0] in activity_ids]
    output = {}
    for date, value in tuple_data:
        if date in output.keys():
            for k, v in value.items():
                output[date][k] += v
        else:
            output[date] = value
    return output


def create_tcx(date: datetime.datetime, distance: int) -> str:
    tcx_string = '<?xml version="1.0" encoding="UTF-8"?>\n'
    tcx_string += '<TrainingCenterDatabase><Activities><Activity Sport="Other">\n'
    tcx_string += '<Id>' + date.strftime("%Y-%m-%dT%H:%M:%SZ") + '</Id>\n'
    tcx_string += '<Lap><TotalTimeSeconds>' + str(int(distance / 100 * 60)) + '</TotalTimeSeconds></Lap>\n'
    tcx_string += '</Activity></Activities></TrainingCenterDatabase>\n'
    return tcx_string


def upload_activity_to_runalyze(tcx_string: str, credentials):
    RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"
    r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads",
                      headers=credentials,
                      files={"file": ("activity.tcx", tcx_string)})
    print(r.text)
    if r.status_code != 201:
        raise Exception("Error uploading activity to Runalyze")


def get_user_confirmation(base_prompt: str, edit_prompt: str) -> tuple[str, str]:
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
    credentials = authorize_with_google(SCOPES, GOOGLE_CREDENTIALS_CACHE_FILE)
    print("Successfully authorized with Google")

    # load credentials for runalyze
    with open(RUNALYZE_CREDENTIALS_FILE, "r", encoding="utf-8") as file:
        credentials_runalyze = json.load(file)

    distance_data_source_id = "derived:com.google.distance.delta:com.google.android.gms:merge_distance_delta"
    activity_summary_data_source_id = "derived:com.google.activity.segment:com.google.android.gms:merge_activity_segments"

    tz_info = datetime.timezone(datetime.timedelta(hours=1), "Europe/Berlin")
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        start_time = datetime.datetime.strptime(file.readline().strip(), "%Y-%m-%d").replace(tzinfo=tz_info)
    # end with today at midnight (as in general there will be additional walking today)
    end_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz_info)

    aggregated_distance_data = get_aggregated_data_from_google_fit(credentials, distance_data_source_id, start_time, end_time)
    distance_data = extract_data_from_aggregated_data(aggregated_distance_data)

    aggregated_activity_data = get_aggregated_data_from_google_fit(credentials, activity_summary_data_source_id, start_time, end_time)
    activity_data = extract_data_from_aggregated_data(aggregated_activity_data)
    biking_data = extract_acitivity_segment_data(activity_data, BIKING_ACTIVITY_IDS)
    walking_data = extract_acitivity_segment_data(activity_data, WALKING_ACTIVITY_IDS)

    for date, distance_list in distance_data.items():
        distance = int(distance_list[0][0])  # use first and only element and cast it to int
        confirmation_str = None
        if date in biking_data:
            confirmation_str = \
                f"On {date.strftime('%Y-%m-%d')} you biked {biking_data[date]['num_segments']} times for a " + \
                f"total of {humanize.precisedelta(datetime.timedelta(milliseconds=biking_data[date]['duration']))}. " + \
                f"Is the total distance of {distance/1000} km correct nevertheless? "
        elif distance > MAX_DISTANCE_WITHOUT_CONFIRMATION:
            confirmation_str = f"On {date.strftime('%Y-%m-%d')} you walked {distance/1000} km. Is this correct? "
        elif date in walking_data:
            pace = (walking_data[date]['duration']/1000/60) / (distance/1000)
            if pace < FASTEST_NORMAL_WALKING_PACE:
                confirmation_str = f"On {date.strftime('%Y-%m-%d')} you walked {distance/1000} km in " + \
                                   humanize.precisedelta(datetime.timedelta(milliseconds=walking_data[date]['duration']),
                                                         format="%.1f", minimum_unit="minutes") + \
                                   f" at {pace:.1f} min/km. Is this correct? "

        # if there was an irregularity, let the user confirm the distance, but only if it is above a certain threshold
        if confirmation_str and distance > MIN_DISTANCE_FOR_CONFIRMATION:
            answer, mode = get_user_confirmation(confirmation_str, "Please enter the correct distance in km: ")
            if mode == "skip":
                continue
            elif mode == "edit":
                distance = int(float(answer)*1000)

        tcx_string = create_tcx(date, distance)
        upload_activity_to_runalyze(tcx_string, credentials_runalyze)

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(end_time.strftime("%Y-%m-%d"))

    print("Done")


if __name__ == "__main__":
    main()
