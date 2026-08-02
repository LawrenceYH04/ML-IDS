"""
web/pipeline.py — thin scoring wrapper around the trained IDSPipeline.

Loads the real-time subset of the ensemble (XGBoost classifier + Autoencoder
anomaly detector; Random Forest is skipped because it needs ~11 GB and ~6 min
per run — unusable for live scoring). Turns a raw CICFlowMeter DataFrame chunk
into a list of per-flow alert events for the dashboard.
"""
import os
# OpenMP guards MUST be set before torch / xgboost are imported (otherwise the
# libiomp5-vs-libomp conflict segfaults the process on macOS). Mirrors inference.py.
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import math
import numpy as np
import pandas as pd

# Reuse the trained pipeline defined in notebooks/inference.py (paths inside it
# are resolved relative to that file, so it works no matter where we run from).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'notebooks'))
from inference import IDSPipeline  # noqa: E402

# Attack severity by class (mirrors explainability.ipynb; default MEDIUM).
_SEVERITY = {
    'SQL Injection':          'CRITICAL',
    'Infilteration':          'CRITICAL',
    'Brute Force -Web':       'CRITICAL',
    'Brute Force -XSS':       'CRITICAL',
    'Bot':                    'HIGH',
    'DDOS attack-HOIC':       'HIGH',
    'DDOS attack-LOIC-UDP':   'HIGH',
    'DDoS attacks-LOIC-HTTP': 'HIGH',
    'DoS attacks-GoldenEye':  'HIGH',
    'DoS attacks-Hulk':       'HIGH',
    'DoS attacks-SlowHTTPTest': 'MEDIUM',
    'DoS attacks-Slowloris':  'MEDIUM',
    'FTP-BruteForce':         'HIGH',
    'SSH-Bruteforce':         'HIGH',
}

_PROTO = {6: 'TCP', 17: 'UDP', 0: 'HOPOPT', 1: 'ICMP'}


def severity_for(cls: str, is_anomaly: bool, lab_attack: bool = False) -> str:
    if cls != 'Benign':
        # lab-detected attacks have no CIC class name → default HIGH
        return _SEVERITY.get(cls, 'HIGH')
    if lab_attack:
        return 'HIGH'
    return 'LOW' if is_anomaly else 'NONE'


def _clean(v):
    """NaN/NaT -> None so json.dumps never emits a bare `NaN` (which the
    browser's JSON.parse rejects, silently freezing the dashboard)."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


class WebScorer:
    """Wraps IDSPipeline and exposes score(df) -> list[event dict]."""

    def __init__(self):
        self.pipe = IDSPipeline()
        self.classes = list(self.pipe.le.classes_)
        self._n = 0

    def score(self, df: pd.DataFrame) -> list:
        df = df.copy()
        df.columns = df.columns.str.strip()

        true_labels = df['Label'].str.strip().values if 'Label' in df.columns else None
        proto = df['Protocol'].values if 'Protocol' in df.columns else None
        # Flow-identity columns exist in CICFlowMeter-V4 output (lab/watch) but not
        # in the released 2018 per-day CSVs (which keep only Dst Port). Extract when
        # present so the dashboard can show source talkers + a per-flow detail view.
        src_ip   = df['Src IP'].values   if 'Src IP'   in df.columns else None
        dst_ip   = df['Dst IP'].values   if 'Dst IP'   in df.columns else None
        src_port = df['Src Port'].values if 'Src Port' in df.columns else None

        X_hybrid, meta_df = self.pipe.preprocess(df)
        preds = self.pipe.predict(X_hybrid, use_rf=False)   # XGB + AE, no RF
        # The lab-native detector is authoritative for LAB traffic but misfires on
        # in-distribution 2018 benign. Set SKIP_LAB=1 to score with the 2018
        # XGB + AE only (used for the in-distribution demo).
        if os.environ.get('SKIP_LAB') == '1':
            n = len(df)
            lab_pred = np.zeros(n, dtype=bool)
            lab_pr   = np.zeros(n, dtype=float)
        else:
            lab_pred, lab_pr = self.pipe.predict_lab(df)    # lab-native attack detector
        out = self.pipe.build_output(meta_df, preds, true_labels,
                                     lab_attack=lab_pred, lab_proba=lab_pr)

        has_ts    = 'Timestamp' in out.columns
        has_port  = 'Dst Port' in out.columns
        has_lab   = 'lab_attack' in out.columns
        has_recon = 'recon_mse' in out.columns
        events = []
        for i in range(len(out)):
            self._n += 1
            row = out.iloc[i]
            cls      = row['predicted_class']
            is_anom  = bool(row['is_anomaly_ae'])
            lab_atk  = bool(row['lab_attack']) if has_lab else False
            # The lab detector is authoritative for lab traffic; when it fires but
            # the 2018 multiclass says Benign, surface it as a lab-detected attack.
            disp_cls = cls
            if lab_atk and cls == 'Benign':
                disp_cls = 'Attack (lab-detected)'
            proto_n = int(proto[i]) if proto is not None and pd.notna(proto[i]) else None
            events.append({
                'id':            self._n,
                'ts':            str(row['Timestamp']) if has_ts and pd.notna(row['Timestamp']) else '',
                'dst_port':      int(row['Dst Port']) if has_port and pd.notna(row['Dst Port']) else None,
                'protocol':      _PROTO.get(proto_n, str(proto_n) if proto_n is not None else '—'),
                'predicted_class': disp_cls,
                'confidence':    round(float(row['confidence']), 4),
                'is_anomaly':    is_anom,
                'recon_mse':     round(float(row['recon_mse']), 6) if has_recon else None,
                'lab_attack':    lab_atk,
                'lab_attack_proba': round(float(row['lab_attack_proba']), 4) if has_lab else None,
                'alert':         bool(row['alert']),
                'severity':      severity_for(disp_cls, is_anom, lab_atk),
                'xgb_class':     _clean(row['xgb_class']) if 'xgb_class' in out.columns else cls,
                'mlp_class':     _clean(row['mlp_class']) if 'mlp_class' in out.columns else '',
                'src_ip':        str(src_ip[i])   if src_ip   is not None and pd.notna(src_ip[i])   else None,
                'dst_ip':        str(dst_ip[i])   if dst_ip   is not None and pd.notna(dst_ip[i])   else None,
                'src_port':      int(src_port[i]) if src_port is not None and pd.notna(src_port[i]) else None,
                'true_label':    _clean(row['true_label']) if true_labels is not None else None,
            })
        return events
