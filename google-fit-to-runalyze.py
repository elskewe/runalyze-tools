from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read"]

credentials = None

credentials_cache_file = "credentials.json"

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
