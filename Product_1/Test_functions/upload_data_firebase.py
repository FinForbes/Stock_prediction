import firebase_admin
from firebase_admin import credentials, storage
import os

# Path to your service account key file
cred = credentials.Certificate('../fin-forbes-dev-firebase-adminsdk-rniyd-c73debf46a.json')

# Initialize the app with your credentials and storage bucket
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fin-forbes-dev.appspot.com'
})


def upload_csv_to_firebase(local_file_path, firebase_file_path):
    bucket = storage.bucket()
    blob = bucket.blob(firebase_file_path)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {firebase_file_path}.")


# Usage
local_csv_path = '../Cloud/script_data'

firebase_path = 'csv_files'
for filename in os.listdir(local_csv_path):
    local_path= local_csv_path+ f'/{filename}'
    server_path= firebase_path+ f'/{filename}'
    upload_csv_to_firebase(local_path, server_path)
