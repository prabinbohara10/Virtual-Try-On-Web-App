# main.py
from flask import Flask
from g_drive_service import GoogleDriveService

app = Flask(__name__)

@app.route('/gdrive-files')
def get_file_list_from_gdrive():
    g_drive_service = GoogleDriveService()
    return {"files": g_drive_service.list_files()}

@app.route('/upload-file')
def upload_file_to_gdrive():
    file_path = 'download.jpg'  # Provide the path to your file
    folder_id = '1DEzVAPGogBIhOmcTY-HBmYDUQlT9epL3'      # Provide the ID of the folder to upload to
    g_drive_service = GoogleDriveService()
    g_drive_service.upload_file(file_path, folder_id)
    return "File uploaded successfully"

if __name__ == '__main__':
    app.run(debug=True)
