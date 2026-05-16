import shutil, os
from huggingface_hub import HfApi

api   = HfApi()
token = os.environ['HF_TOKEN']
repo  = 'abhisekbasu/predictive-maintenance-model'

shutil.make_archive('mlruns', 'zip', '.', 'mlruns')

files = [
    ('best_engine_model_v1.joblib', 'best_engine_model_v1.joblib'),
    ('model_summary.json',          'model_summary.json'),
    ('mlruns.zip',                  'mlruns/mlruns.zip'),
]
for local, remote in files:
    api.upload_file(
        path_or_fileobj=local, path_in_repo=remote,
        repo_id=repo, repo_type='model', token=token
    )
    print(f'Uploaded: {remote}')
print('Model registered on HF Model Hub')
