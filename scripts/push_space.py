import os
from huggingface_hub import HfApi

api   = HfApi()
token = os.environ['HF_TOKEN']
repo  = 'abhisekbasu/predictive-maintenance-app'

files = [
    ('deployment/app.py',           'app.py'),
    ('deployment/requirements.txt', 'requirements.txt'),
    ('deployment/Dockerfile',       'Dockerfile'),
]
for local, remote in files:
    api.upload_file(
        path_or_fileobj=local, path_in_repo=remote,
        repo_id=repo, repo_type='space', token=token
    )
    print(f'Pushed: {remote}')
print('Deployment pushed to HF Spaces')
