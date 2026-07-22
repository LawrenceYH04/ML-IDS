"""RF inference subprocess — called by inference.py via subprocess.

Usage (internal):
    python predict_rf_live.py <input_npy> <output_npy>

Loads RF from models/, predicts on the numpy array at <input_npy>,
saves (y_pred, proba) stacked to <output_npy>, then exits so the OS
reclaims the ~11 GB RF memory before the parent process continues.
"""
import sys, os
import numpy as np
import joblib

if len(sys.argv) != 3:
    print("Usage: predict_rf_live.py <input_npy> <output_npy>", file=sys.stderr)
    sys.exit(1)

INPUT_PATH  = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
MODELS      = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

def predict_calibrated(proba, thresholds):
    masked     = np.where(proba >= thresholds, proba, -1.0)
    no_qualify = np.all(masked < 0, axis=1)
    result     = np.argmax(masked, axis=1)
    result[no_qualify] = np.argmax(proba[no_qualify], axis=1)
    return result

print("  RF subprocess: loading X_hybrid ...", flush=True)
X = np.load(INPUT_PATH)

print("  RF subprocess: loading random_forest.pkl (~11 GB expanded in memory) ...", flush=True)
rf         = joblib.load(os.path.join(MODELS, 'random_forest.pkl'))
thresholds = np.load(os.path.join(MODELS, 'thresholds_rf.npy'))

print("  RF subprocess: predicting ...", flush=True)
proba_rf = rf.predict_proba(X)
y_pred   = predict_calibrated(proba_rf, thresholds)

# Save y_pred and proba_rf separately — avoids making a full in-memory copy of
# proba_rf (43 MB) just to stack it. The proba path must end in '.npy' so that
# np.save does not append a second '.npy' (which left the file where the parent
# could not find it — the RF branch then silently fell back to XGB-only).
# Parent derives the same path via the identical rule below.
PROBA_PATH = (OUTPUT_PATH[:-4] if OUTPUT_PATH.endswith('.npy') else OUTPUT_PATH) + '_proba.npy'
np.save(OUTPUT_PATH, y_pred)
np.save(PROBA_PATH,  proba_rf)
print(f"  RF subprocess: done — saved to {OUTPUT_PATH}", flush=True)
# Process exits here — OS reclaims all ~11 GB
