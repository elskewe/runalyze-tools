import datetime
import json
import pprint

import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
import googleapiclient.discovery

def authorize_with_google(SCOPES, credentials_cache_file):
   credentials = None
   # Check if credentials are already cached
   if os.path.exists(credentials_cache_file):
      credentials = Credentials.from_authorized_user_file(
      credentials_cache_file, SCOPES)
   
   # if credentials are not cached or expired, ask for user to log in
   if not credentials or not credentials.valid:
      if credentials and credentials.expired and credentials.refresh_token:
         credentials.refresh(Request())
      else:
         flow = InstalledAppFlow.from_client_secrets_file(
         "client_secret.json", SCOPES)
         credentials = flow.run_local_server(port=8080)
   
      # save credentials to file
      with open(credentials_cache_file, "w") as file:
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

def extract_distance_data_from_aggregated_data(aggregated_data):
   distance_data = {}
   for bucket in aggregated_data['bucket']:
      dataset = bucket['dataset']
      if len(dataset) > 1:
         raise Exception("More than one point in dataset")
      # get date by using the average of the start and end time which should be the right date for timezones around utc
      date = datetime.datetime.fromtimestamp(
         (int(bucket['startTimeMillis']) + int(bucket['endTimeMillis'])) / 2 / 1000)
      if date in distance_data.keys():
         raise Exception("Duplicate date in dataset")
      point = dataset[0]['point']
      if len(point) == 0:
         continue
      distance_data[date] = int(point[0]['value'][0]['fpVal'])
   return distance_data

def create_tcx(date, distance):
   tcx_string = '<?xml version="1.0" encoding="UTF-8"?>\n'
   tcx_string += '<TrainingCenterDatabase><Activities><Activity Sport="Other">\n'
   tcx_string += '<Id>' + date.strftime("%Y-%m-%dT%H:%M:%SZ") + '</Id>\n'
   tcx_string += '<Lap><TotalTimeSeconds>' + str(int(distance / 100 * 60)) + '</TotalTimeSeconds></Lap>\n'
   tcx_string += '</Activity></Activities></TrainingCenterDatabase>\n'
   return tcx_string

def upload_activity_to_runalyze(tcx_string, credentials):
   RUNALYZE_API_ENDPOINT = "https://runalyze.com/api/v1/"
   r = requests.post(RUNALYZE_API_ENDPOINT + "activities/uploads",
                     headers=credentials,
                     files={"file": ("activity.tcx", tcx_string)})
   print(r.text)
   if r.status_code != 201:
      raise Exception("Error uploading activity to Runalyze")


SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read", 
          "https://www.googleapis.com/auth/fitness.location.read"]
credentials_cache_file = "credentials.json"

credentials = authorize_with_google(SCOPES, credentials_cache_file)
print("Successfully authorized with Google")

# load credentials for runalyze
with open("runalyze_credentials.json", "r") as file:
   credentials_runalyze = json.load(file)

data_source_id = "derived:com.google.distance.delta:com.google.android.gms:merge_distance_delta"

tz_info = datetime.timezone(datetime.timedelta(hours=1), "Europe/Berlin")
with open("config.txt", "r") as file:
   start_time = datetime.datetime.strptime(file.readline().strip(), "%Y-%m-%d").replace(tzinfo=tz_info)
# for testing purposes, start with two days later than the start time
end_time = start_time + datetime.timedelta(days=2)

aggregated_data = get_aggregated_data_from_google_fit(credentials, data_source_id, start_time, end_time)

distance_data = extract_distance_data_from_aggregated_data(aggregated_data)

for date, distance in distance_data.items():
   if distance > 8000:
      while True:
         # ask user to confirm the distance, give the correct distance or skip the day
         answer = input("On " + date.strftime("%Y-%m-%d") + " you walked "
                        + str(distance/1000) + " km. Is this correct? yes/skip/edit: ")
         if answer == "yes":
            break
         elif answer == "skip":
            break
         elif answer == "edit":
            distance = float(input("Please enter the correct distance in km: "))*1000
            break
         else:
            print("Please enter yes, skip or edit")
            continue
      if answer == "skip":
         continue
      
   tcx_string = create_tcx(date, distance)
   upload_activity_to_runalyze(tcx_string, credentials_runalyze)

with open("config.txt", "w") as file:
   file.write(end_time.strftime("%Y-%m-%d"))

print("Done")
