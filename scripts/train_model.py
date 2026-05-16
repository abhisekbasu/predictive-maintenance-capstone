import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')
TARGET       = 'Engine_Condition'
FEATURE_COLS = [c for c in train_df.columns if c != TARGET]
X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET]

mlflow.set_tracking_uri('file:./mlruns')
mlflow.set_experiment('predictive_maintenance_pipeline')

with mlflow.start_run(run_name='Pipeline_RandomForest'):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        max_features='sqrt', min_samples_leaf=1,
        min_samples_split=2, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy'  : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred),
        'recall'    : recall_score(y_test, y_pred),
        'f1'        : f1_score(y_test, y_pred),
        'roc_auc'   : roc_auc_score(y_test, y_prob)
    }
    mlflow.log_params({
        'n_estimators': 300, 'max_depth': 'None',
        'max_features': 'sqrt', 'min_samples_leaf': 1,
        'min_samples_split': 2, 'class_weight': 'balanced'
    })
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, 'model')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

joblib.dump(model, 'best_engine_model_v1.joblib')
summary = {
    'best_model': 'RandomForest',
    'test_metrics': {k: float(v) for k, v in metrics.items()},
    'feature_cols': FEATURE_COLS,
    'training_records': len(X_train),
    'test_records': len(X_test)
}
with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print('Model trained and saved successfully')
