import datetime
import pprint
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

SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read", 
          "https://www.googleapis.com/auth/fitness.location.read"]
credentials_cache_file = "credentials.json"

credentials = authorize_with_google(SCOPES, credentials_cache_file)
print("Successfully authorized with Google")

data_source_id = "derived:com.google.distance.delta:com.google.android.gms:merge_distance_delta"
tz_info = datetime.timezone(datetime.timedelta(hours=1), "Europe/Berlin")
start_time = datetime.datetime(2023, 8, 12, 0, 0, tzinfo=tz_info)
end_time = datetime.datetime(2023, 8, 14, 0, 0, tzinfo=tz_info)

aggregated_data = get_aggregated_data_from_google_fit(credentials, data_source_id, start_time, end_time)

pp = pprint.PrettyPrinter(indent=1)
pp.pprint(aggregated_data)

print("Done")
