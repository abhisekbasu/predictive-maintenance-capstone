import pandas as pd
from huggingface_hub import hf_hub_download
import os

token = os.environ['HF_TOKEN']
repo  = 'abhisekbasu/predictive-maintenance-engine'

train_path = hf_hub_download(
    repo_id=repo, filename='processed/train.csv',
    repo_type='dataset', token=token, force_download=True
)
test_path = hf_hub_download(
    repo_id=repo, filename='processed/test.csv',
    repo_type='dataset', token=token, force_download=True
)
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv',   index=False)
print(f'Train: {train_df.shape}, Test: {test_df.shape}')
