"""
Compute XGBoost feature importances and alert-sample probabilities
in an isolated subprocess — called from explainability.ipynb Cell 4.

Running in a subprocess guarantees the OS reclaims all XGBoost memory
on exit before the main notebook continues, preventing OOM crashes when
the RF model (~11 GB) is already resident in the parent process.
"""
from pathlib import Path
import numpy as np
import joblib

# Resolve paths from this file's location (…/ML-IDS/notebooks/), so the
# script works no matter what the caller's working directory is.
ROOT    = Path(__file__).resolve().parent.parent
MODELS  = str(ROOT / 'models')
RESULTS = str(ROOT / 'results')

print('Loading XGBoost...')
xgb = joblib.load(f'{MODELS}/xgboost.pkl')

# Gain-based feature importances — fast, no SHAP tree traversal needed
# (XGBoost has 45,000 internal trees; full TreeSHAP would take hours)
np.save(f'{RESULTS}/importances_xgb.npy', xgb.feature_importances_)
print('XGBoost feature importances saved.')

# Pre-compute probabilities for the 5 alert samples so Cell 7 can
# apply calibrated thresholds without reloading XGBoost into main process
alert_samples   = np.load(f'{RESULTS}/alert_samples.npy')
alert_xgb_proba = xgb.predict_proba(alert_samples)
np.save(f'{RESULTS}/alert_xgb_proba.npy', alert_xgb_proba)
print('XGBoost alert probabilities saved.')
print('Done — subprocess exiting, OS will reclaim XGBoost memory.')
