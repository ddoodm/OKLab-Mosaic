import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import requests

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']


def get_google_photos_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_80535563940-cscbf3v34v8vbuhpi5e2i42g7btubsa1.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('photoslibrary', 'v1', credentials=creds, static_discovery=False)


def list_albums():
    service = get_google_photos_service()
    results = service.albums().list(pageSize=50).execute()  # Maximum allowed by the API is 50
    albums = results.get('albums', [])
    for album in albums:
        print(f"Album title: {album['title']}, Album ID: {album['id']}")


# Floriasundays:
# AF39uqvR-IHGPp85orXPKuyiRjkHWuH_fiwYuy8knLbg9NoHlcAv89k2iMwHzP2P9u_0lIzZru6M


def download_photos(album_id, download_dir):
    service = get_google_photos_service()

    request_body = {
        'albumId': album_id,
        'pageSize': 100  # Maximum allowed by the API
    }

    while True:
        results = service.mediaItems().search(body=request_body).execute()
        media_items = results.get('mediaItems', [])
        for media_item in media_items:
            mime_type = media_item['mimeType']
            if not mime_type.startswith('image/'):
                print(f'Skipping {media_item["filename"]} as it is not an image')
                continue  # Skip videos

            url = media_item['baseUrl'] + '=w800'
            response = requests.get(url)
            file_name = media_item['filename']
            full_file_path = os.path.join(download_dir, file_name)  # Construct the full path to the file
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)  # Create directories if they don't exist
            with open(full_file_path, 'wb') as file:
                file.write(response.content)
            print(f'Downloaded {full_file_path}')

        # Check for a nextPageToken in the API response
        next_page_token = results.get('nextPageToken')
        if next_page_token:
            request_body['pageToken'] = next_page_token
            print('Got next page token')
        else:
            print('No next page')
            break


download_photos('AF39uqvR-IHGPp85orXPKuyiRjkHWuH_fiwYuy8knLbg9NoHlcAv89k2iMwHzP2P9u_0lIzZru6M', 'floriasundays')
