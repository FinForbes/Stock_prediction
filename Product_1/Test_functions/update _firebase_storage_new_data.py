import firebase_admin
from firebase_admin import credentials, storage
import os
from datetime import datetime, timezone

cred = credentials.Certificate('../fin-forbes-dev-firebase-adminsdk-rniyd-c73debf46a.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fin-forbes-dev.appspot.com'
})


def upload_file(local_file_path, storage_path):
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {storage_path}")


def update_firebase_storage_files(local_folder_path, firebase_folder_path):
    bucket = storage.bucket()

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            if file.endswith('.csv'):
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                storage_path = os.path.join(firebase_folder_path, relative_path)

                local_modified_time = datetime.fromtimestamp(os.path.getmtime(local_file_path), tz=timezone.utc)

                blob = bucket.blob(storage_path)

                try:
                    firebase_modified_time = blob.updated
                    if local_modified_time > firebase_modified_time:
                        upload_file(local_file_path, storage_path)
                except Exception:
                    # File doesn't exist in Firebase Storage, upload it
                    upload_file(local_file_path, storage_path)


local_folder_path = '../Cloud/script_data'
firebase_folder_path = 'csv_files/'

update_firebase_storage_files(local_folder_path, firebase_folder_path)
