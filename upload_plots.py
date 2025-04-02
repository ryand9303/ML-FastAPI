import os
import base64
import requests

# GitHub repo details
GITHUB_TOKEN = ''
GITHUB_USER = 'Emily-M-C'
REPO_NAME = 'ML-FastAPI'
BRANCH = 'main'  # or 'master'
FOLDER_PATH = 'Plots'  # The folder where your plot files are located


# GitHub API URL
GITHUB_API_URL = f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/Plots/'

# Directory where your plots are stored
local_directory = r'C:\Users\anakl\Downloads\hist'  

# Create headers for authentication
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

# Loop through each file in the local directory
for filename in os.listdir(local_directory):
    file_path = os.path.join(local_directory, filename)
    
    if os.path.isfile(file_path):  # Make sure it's a file
        with open(file_path, 'rb') as f:
            content = f.read()

        # Encode file content to base64 (required by GitHub API)
        content_base64 = base64.b64encode(content).decode('utf-8')

        # Create the API payload for the file
        payload = {
            'message': f'Add {filename}',  # Commit message
            'branch': BRANCH_NAME,
            'content': content_base64
        }

        # Make the API request to upload the file
        response = requests.put(
            f'{GITHUB_API_URL}{filename}', 
            headers=headers, 
            json=payload
        )

        if response.status_code == 201:
            print(f'Successfully uploaded {filename} to the repo.')
        else:
            print(f'Failed to upload {filename}. Response: {response.json()}')
