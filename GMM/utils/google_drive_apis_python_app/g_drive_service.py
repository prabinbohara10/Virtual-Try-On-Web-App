# g_drive_service.py
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials

class GoogleDriveService:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive']
        _base_path = os.path.dirname(__file__)
        _credential_path = os.path.join(_base_path, 'credential.json')
        self.credentials = Credentials.from_service_account_file(_credential_path, scopes=self.SCOPES)
        self.service = build('drive', 'v3', credentials=self.credentials)

    def list_files(self, folder_id=None):
        query = "'{}' in parents".format(folder_id) if folder_id else "trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        return results.get("files", [])

    def delete_files_in_folder(self, folder_id):
        files = self.list_files(folder_id)
        for file in files:
            self.service.files().delete(fileId=file['id']).execute()

    def upload_folders(self, folders_path_list, parent_folder_id=None):
        for folder_path in folders_path_list:
            folder_name = os.path.basename(folder_path)
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_folder_id:
                folder_metadata['parents'] = [parent_folder_id]
            folder = self.service.files().create(body=folder_metadata, fields='id').execute()

            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    self.upload_folders([item_path], folder['id'])  # Recursively upload subfolders
                else:
                    media = MediaFileUpload(item_path, resumable=True)
                    file_metadata = {
                        'name': item,
                        'parents': [folder['id']]
                    }
                    file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                    print('File ID:', file.get('id'))
                    

    def main_upload_folders(self, folders_path_list, parent_folder_id=None, bool_delete=True):
        if bool_delete:
            
            self.delete_files_in_folder(parent_folder_id)
        
        self.upload_folders(folders_path_list, parent_folder_id)