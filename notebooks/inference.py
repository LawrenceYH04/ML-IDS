"""
inference.py — ML-IDS live inference pipeline
=============================================
Usage:
    # Score a single CICFlowMeter CSV:
    python inference.py --input flows.csv

    # Score all CSVs in a directory, skip RF to save memory:
    python inference.py --input capture_dir/ --output alerts.csv --skip-rf

    # If your CSV already has a Label column (ground-truth testing):
    python inference.py --input labeled.csv --output alerts.csv

Input:  CSV file(s) produced by CICFlowMeter (77-column format).
        A 'Label' column is optional — used for accuracy reporting only.
Output: CSV with original identifier columns plus:
            predicted_class   — ensemble IDS classification
            confidence        — max class probability (0–1)
            is_anomaly_ae     — 1 if AE reconstruction error > threshold
            alert             — 1 if predicted attack OR is_anomaly_ae
            xgb_class         — XGBoost class (for debugging)
            mlp_class         — MLP class (for debugging)
            rf_class          — RF class (only if --skip-rf not set)
"""
import os
# Must be set before any torch / xgboost import to prevent OpenMP library
# conflict (PyTorch libiomp5 vs XGBoost libomp) that causes a segfault on macOS.
os.environ.setdefault('OMP_NUM_THREADS',     '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS',     '1')

import argparse, glob, json, subprocess, sys, tempfile, time, gc
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ── Paths (relative to this script's location) ────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(_HERE, '..', 'models')
RAW_DIR = os.path.join(_HERE, '..', 'data', 'raw')

# Columns that identify the flow but are not features (same set as preprocessing)
_ID_COLS = {'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Label'}
# Columns to keep in the output for context
_META_COLS = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
# TCP flag columns that the training data stores as binary 0/1 presence flags
# but the official Java CICFlowMeter emits as true per-flow counts. Binarized in
# preprocess() so live captures match the training encoding. Names are the
# stripped training-schema names (post column-rename).
_FLAG_COUNT_COLS = ['FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
                    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt']

# Official Java CICFlowMeter (V4) emits fuller column names than the CIC-IDS-2018
# training schema. Map V4 → training names so live captures from the x86 extractor
# align to feature_cols. A no-op on data already in 2018 naming (keys simply absent),
# so it is safe to apply unconditionally to any input.
_V4_RENAME = {
    'Total Fwd Packet': 'Tot Fwd Pkts', 'Total Bwd packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packet': 'TotLen Fwd Pkts', 'Total Length of Bwd Packet': 'TotLen Bwd Pkts',
    'Fwd Packet Length Max': 'Fwd Pkt Len Max', 'Fwd Packet Length Min': 'Fwd Pkt Len Min',
    'Fwd Packet Length Mean': 'Fwd Pkt Len Mean', 'Fwd Packet Length Std': 'Fwd Pkt Len Std',
    'Bwd Packet Length Max': 'Bwd Pkt Len Max', 'Bwd Packet Length Min': 'Bwd Pkt Len Min',
    'Bwd Packet Length Mean': 'Bwd Pkt Len Mean', 'Bwd Packet Length Std': 'Bwd Pkt Len Std',
    'Flow Bytes/s': 'Flow Byts/s', 'Flow Packets/s': 'Flow Pkts/s',
    'Fwd IAT Total': 'Fwd IAT Tot', 'Bwd IAT Total': 'Bwd IAT Tot',
    'Fwd Header Length': 'Fwd Header Len', 'Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s', 'Bwd Packets/s': 'Bwd Pkts/s',
    'Packet Length Min': 'Pkt Len Min', 'Packet Length Max': 'Pkt Len Max',
    'Packet Length Mean': 'Pkt Len Mean', 'Packet Length Std': 'Pkt Len Std',
    'Packet Length Variance': 'Pkt Len Var',
    'FIN Flag Count': 'FIN Flag Cnt', 'SYN Flag Count': 'SYN Flag Cnt',
    'RST Flag Count': 'RST Flag Cnt', 'PSH Flag Count': 'PSH Flag Cnt',
    'ACK Flag Count': 'ACK Flag Cnt', 'URG Flag Count': 'URG Flag Cnt',
    'CWR Flag Count': 'CWE Flag Count', 'ECE Flag Count': 'ECE Flag Cnt',
    'Average Packet Size': 'Pkt Size Avg', 'Fwd Segment Size Avg': 'Fwd Seg Size Avg',
    'Bwd Segment Size Avg': 'Bwd Seg Size Avg',
    'Fwd Bytes/Bulk Avg': 'Fwd Byts/b Avg', 'Fwd Packet/Bulk Avg': 'Fwd Pkts/b Avg',
    'Fwd Bulk Rate Avg': 'Fwd Blk Rate Avg', 'Bwd Bytes/Bulk Avg': 'Bwd Byts/b Avg',
    'Bwd Packet/Bulk Avg': 'Bwd Pkts/b Avg', 'Bwd Bulk Rate Avg': 'Bwd Blk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts', 'Subflow Fwd Bytes': 'Subflow Fwd Byts',
    'Subflow Bwd Packets': 'Subflow Bwd Pkts', 'Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'FWD Init Win Bytes': 'Init Fwd Win Byts', 'Bwd Init Win Bytes': 'Init Bwd Win Byts',
}


# ── Model architecture definitions (must match training notebooks exactly) ──

class Autoencoder(nn.Module):
    """Feature-extraction AE: 77 → 64 → 32 → 16 → 32 → 64 → 77.

    MUST match feature_engineering.ipynb exactly: LeakyReLU(0.01) hidden layers
    and a LINEAR bottleneck (no activation on the 16-dim latent). The training
    notebook deliberately switched away from ReLU to stop latent collapse and to
    keep the latent signed/full-range. Using ReLU here (esp. a terminal ReLU on
    the bottleneck) hard-zeros every negative latent dim, so the latent_scaler
    (fitted on signed encodings) mis-scales the 16 latent features, the 56-dim
    hybrid vector lands off-manifold, and the anomaly AE then flags ~everything.
    Tree models tolerate it (structure survives) which is why only the AE broke.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32),        nn.LeakyReLU(0.01),
            nn.Linear(32, 16),                              # linear bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class AEAnomaly(nn.Module):
    """Anomaly-detection AE trained on benign-only: 56 → 32 → 16 → 32 → 56.

    Retrained on official-CICFlowMeter LAB benign (build/retrain_anomaly.py).
    LeakyReLU hidden layers and a LINEAR output: the AE now sees RobustScaler-
    normalised hybrid features (self.ae_input_scaler) whose range is unbounded,
    so a Sigmoid output (capped [0,1]) can no longer reconstruct them. Must stay
    identical to the AEAnomaly class in build/retrain_anomaly.py.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(56, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 16), nn.LeakyReLU(0.01),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 56),                       # linear output
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class MLP(nn.Module):
    """Classification MLP: 56 → 512 → 256 → 128 → 15."""
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(56, 512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.network(x)


def _predict_calibrated(proba, thresholds):
    masked     = np.where(proba >= thresholds, proba, -1.0)
    no_qualify = np.all(masked < 0, axis=1)
    result     = np.argmax(masked, axis=1)
    result[no_qualify] = np.argmax(proba[no_qualify], axis=1)
    return result


# ── Pipeline class ─────────────────────────────────────────────────────────

class IDSPipeline:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps'  if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f'Device : {self.device}')

        # ── Preprocessing artifacts ──────────────────────────────────────
        self.imputer       = joblib.load(os.path.join(MODELS, 'imputer.pkl'))
        self.scaler        = joblib.load(os.path.join(MODELS, 'scaler.pkl'))
        self.latent_scaler = joblib.load(os.path.join(MODELS, 'latent_scaler.pkl'))
        self.le            = joblib.load(os.path.join(MODELS, 'label_encoder.pkl'))
        self.selected_idx  = np.load(os.path.join(MODELS, 'selected_indices.npy'))

        # ── Feature-extraction AE ────────────────────────────────────────
        input_dim = len(self.imputer.statistics_)   # 77
        self.feat_ae = Autoencoder(input_dim).to(self.device)
        self.feat_ae.load_state_dict(
            torch.load(os.path.join(MODELS, 'autoencoder.pth'),
                       map_location=self.device, weights_only=True))
        self.feat_ae.eval()

        # ── Anomaly-detection AE ─────────────────────────────────────────
        self.ae_anomaly = AEAnomaly().to(self.device)
        self.ae_anomaly.load_state_dict(
            torch.load(os.path.join(MODELS, 'autoencoder_anomaly.pth'),
                       map_location=self.device, weights_only=True))
        self.ae_anomaly.eval()
        self.ae_threshold = float(
            np.load(os.path.join(MODELS, 'threshold_ae_anomaly.npy')).flat[0])
        # Anomaly-only input scaler fitted on lab benign (build/retrain_anomaly.py).
        # Applied to the 56-dim hybrid before the anomaly AE only — the classifier
        # path is unaffected. Absent → identity (pre-lab-retrain checkpoints).
        _asc = os.path.join(MODELS, 'ae_input_scaler.pkl')
        self.ae_input_scaler = joblib.load(_asc) if os.path.exists(_asc) else None

        # ── Lab-native attack detector (build/retrain_lab_classifier.py) ──
        # Binary benign-vs-attack XGBoost on RAW CICFlowMeter features. Trees are
        # scale-invariant, so this works on live lab flows where the 2018-fit
        # hybrid pipeline makes the multiclass classifier predict Benign for
        # everything. Absent → None (falls back to the 2018-only alert logic).
        _lab = os.path.join(MODELS, 'xgb_lab_binary.pkl')
        if os.path.exists(_lab):
            self.lab_clf     = joblib.load(_lab)
            self.lab_imputer = joblib.load(os.path.join(MODELS, 'lab_imputer.pkl'))
            self.lab_cols    = json.load(open(os.path.join(MODELS, 'lab_feature_cols.json')))
            print(f'Lab detector: loaded ({len(self.lab_cols)} raw features)')
        else:
            self.lab_clf = self.lab_imputer = self.lab_cols = None

        # ── MLP ──────────────────────────────────────────────────────────
        self.mlp = MLP(len(self.le.classes_)).to(self.device)
        self.mlp.load_state_dict(
            torch.load(os.path.join(MODELS, 'mlp.pth'),
                       map_location=self.device, weights_only=True))
        self.mlp.eval()

        # ── XGBoost ──────────────────────────────────────────────────────
        self.xgb = joblib.load(os.path.join(MODELS, 'xgboost.pkl'))

        # ── Thresholds ───────────────────────────────────────────────────
        self.t_mlp      = np.load(os.path.join(MODELS, 'thresholds_mlp.npy'))
        self.t_xgb      = np.load(os.path.join(MODELS, 'thresholds_xgb.npy'))
        self.t_rf       = np.load(os.path.join(MODELS, 'thresholds_rf.npy'))
        self.t_ensemble = 0.5 * self.t_rf + 0.5 * self.t_xgb

        # ── Expected feature column order (reconstructed from raw CSV headers) ──
        self.feature_cols = self._build_feature_cols()

        n_cls = len(self.le.classes_)
        print(f'Classes: {list(self.le.classes_)}')
        print(f'AE anomaly threshold: {self.ae_threshold:.6f}')
        print(f'Ready — {input_dim} input features → 56 hybrid features → {n_cls} classes\n')

    def _build_feature_cols(self):
        """Return the 77 feature column names in training order.

        Prefers the canonical list saved by preprocessing.ipynb
        (models/feature_cols.json). Falls back to reconstructing it from the
        raw CSV headers, deduplicating on stripped names so inconsistent header
        whitespace across files cannot silently misalign the column order.
        """
        canonical = os.path.join(MODELS, 'feature_cols.json')
        if os.path.exists(canonical):
            with open(canonical) as f:
                return json.load(f)
        raw_csvs = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv')))
        if not raw_csvs:
            print('WARNING: models/feature_cols.json and data/raw/*.csv both '
                  'missing — column alignment disabled.')
            print('         Make sure your input CSVs have columns in the same order'
                  ' as the training data.')
            return None
        seen, cols = set(), []
        for f in raw_csvs:
            for c in pd.read_csv(f, nrows=0).columns:
                cs = c.strip()
                if cs in _ID_COLS or cs in seen:
                    continue
                seen.add(cs)
                cols.append(cs)
        return cols

    # ── Core transform ────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame):
        """
        Raw CICFlowMeter DataFrame → 56-dim hybrid feature array.

        Steps mirror preprocessing.ipynb + feature_engineering.ipynb exactly:
          1. Drop identifier/label columns
          2. Coerce to numeric (handles Infinity → NaN)
          3. Clip ±1e15 before imputation to prevent overflow in scaler
          4. Impute (median), then MinMaxScale
          5. Select 40 MI features
          6. Encode 77-dim input through feature AE → 16 latent dims
          7. Scale latent dims, concatenate → 56 hybrid features
        """
        # Keep a copy of identifier columns for the output report
        meta_cols = [c for c in df.columns if c.strip() in set(_META_COLS)]
        meta_df   = df[meta_cols].copy().reset_index(drop=True)

        # Drop identifiers and label
        drop  = [c for c in df.columns if c.strip() in _ID_COLS]
        X     = df.drop(columns=drop, errors='ignore').copy()
        X.columns = X.columns.str.strip()

        # Map official-CICFlowMeter (V4) headers to the training schema. No-op if
        # the input already uses 2018 names (keys absent), so both sources work.
        X = X.rename(columns=_V4_RENAME)

        # Coerce to numeric; replace Inf with NaN so the imputer fills
        # them with training medians rather than passing huge clipped values
        # through the MinMaxScaler (which would produce out-of-range [0,1] outputs).
        X = X.apply(pd.to_numeric, errors='coerce')
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Binarize TCP flag "counts" to match the training encoding.
        # The CIC-IDS-2018 training CSVs store these columns as binary presence
        # indicators (0/1) — verified: max=1 across all 1.05M rows, even for
        # 9000-packet flows. The official Java CICFlowMeter instead emits true
        # per-flow counts (e.g. ACK up to 24). Feeding raw counts through the
        # scaler (fitted on 0/1) yields wildly out-of-range values and inflates
        # the anomaly AE's reconstruction error. Clipping to {0,1} realigns the
        # encoding; it is a no-op on the already-binary training data.
        for _c in _FLAG_COUNT_COLS:
            if _c in X.columns:
                X[_c] = (X[_c].fillna(0) > 0).astype(np.float32)

        # Align columns to training order
        if self.feature_cols is not None:
            X = X.reindex(columns=self.feature_cols)

        # Validate column count before imputer — gives a clear error instead of
        # a cryptic sklearn shape mismatch if the CSV has the wrong structure.
        expected = len(self.imputer.statistics_)
        if X.shape[1] != expected:
            raise ValueError(
                f'Input has {X.shape[1]} feature columns after dropping identifiers; '
                f'expected {expected}. Check that your CSV is in CICFlowMeter format '
                f'and that the raw training CSVs are available in data/raw/ for '
                f'column alignment.')

        # Impute → scale (transforms fitted on training data)
        # Pass numpy array to suppress sklearn "fitted without feature names" warning
        X_imp    = self.imputer.transform(X.to_numpy())
        X_scaled = self.scaler.transform(X_imp)

        # 40 MI-selected handcrafted features
        X_hand = X_scaled[:, self.selected_idx]

        # 16 latent features from feature-extraction AE
        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        latent_chunks = []
        with torch.no_grad():
            for s in range(0, len(X_t), 4096):
                latent_chunks.append(
                    self.feat_ae.encoder(X_t[s:s + 4096]).cpu().numpy())
        X_latent = self.latent_scaler.transform(np.concatenate(latent_chunks))

        return np.concatenate([X_hand, X_latent], axis=1).astype(np.float32), meta_df

    # ── Model inference ───────────────────────────────────────────────────
    def _mlp_proba(self, X):
        X_t, chunks = torch.from_numpy(X).to(self.device), []
        with torch.no_grad():
            for s in range(0, len(X_t), 4096):
                logits = self.mlp(X_t[s:s + 4096])
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(chunks)

    def _xgb_proba(self, X):
        chunks = []
        for s in range(0, len(X), 50_000):
            chunks.append(self.xgb.predict_proba(X[s:s + 50_000]))
        return np.concatenate(chunks)

    def _ae_anomaly_mse(self, X):
        # Normalise with the anomaly-only scaler so the linear-output AE sees the
        # same distribution it was trained on (no-op if the scaler is absent).
        if self.ae_input_scaler is not None:
            X = self.ae_input_scaler.transform(X).astype(np.float32)
        X_t, chunks = torch.from_numpy(X).to(self.device), []
        with torch.no_grad():
            for s in range(0, len(X_t), 4096):
                b    = X_t[s:s + 4096]
                r    = self.ae_anomaly(b)
                chunks.append(((b - r) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(chunks)

    def predict(self, X_hybrid, use_rf=True):
        """
        Run all models on 56-dim hybrid features.

        Returns a dict with arrays aligned to X_hybrid rows.
        If use_rf=True, RF is run in a subprocess (avoids ~11 GB in-process load).
        """
        proba_mlp = self._mlp_proba(X_hybrid)
        proba_xgb = self._xgb_proba(X_hybrid)
        recon_mse = self._ae_anomaly_mse(X_hybrid)

        y_mlp = _predict_calibrated(proba_mlp, self.t_mlp)
        y_xgb = _predict_calibrated(proba_xgb, self.t_xgb)

        proba_rf, y_rf = None, None
        if use_rf:
            proba_rf, y_rf = self._run_rf_subprocess(X_hybrid)

        # Ensemble: RF+XGB if RF available, else XGB only
        if proba_rf is not None:
            ensemble_proba = 0.5 * proba_rf + 0.5 * proba_xgb
            y_ensemble     = _predict_calibrated(ensemble_proba, self.t_ensemble)
            confidence     = ensemble_proba.max(axis=1)
        else:
            y_ensemble = y_xgb
            confidence = proba_xgb.max(axis=1)

        return {
            'y_ensemble':  y_ensemble,
            'y_rf':        y_rf,
            'y_xgb':       y_xgb,
            'y_mlp':       y_mlp,
            'confidence':  confidence,
            'recon_mse':   recon_mse,
            'is_anomaly':  (recon_mse > self.ae_threshold).astype(int),
        }

    def predict_lab(self, df):
        """Lab-native binary attack prediction from the RAW CICFlowMeter df.

        Returns (pred 0/1, attack-probability), or (None, None) if the lab
        detector isn't installed. Uses the raw V4 feature columns directly — no
        rename/scaling — matching how the lab XGBoost was trained.
        """
        if self.lab_clf is None:
            return None, None
        X = df.copy(); X.columns = X.columns.str.strip()
        X = (X.reindex(columns=self.lab_cols).apply(pd.to_numeric, errors='coerce')
             .replace([np.inf, -np.inf], np.nan).to_numpy(dtype='float64'))
        Xi = self.lab_imputer.transform(X)
        proba = self.lab_clf.predict_proba(Xi)[:, 1]
        return (proba >= 0.5).astype(int), proba

    def _run_rf_subprocess(self, X_hybrid):
        """Run RF in an isolated subprocess; return (proba_rf, y_rf)."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fin:
            tmp_in = fin.name
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fout:
            tmp_out = fout.name
        # Must match predict_rf_live.py's PROBA_PATH derivation exactly.
        tmp_proba = (tmp_out[:-4] if tmp_out.endswith('.npy') else tmp_out) + '_proba.npy'
        try:
            np.save(tmp_in, X_hybrid)
            script = os.path.join(_HERE, 'predict_rf_live.py')
            # Let stdout flow to the terminal in real-time so RF progress messages
            # are visible during the ~6-min inference run. Only capture stderr for
            # error detection. Timeout after 20 min to avoid hanging on OOM kill.
            result = subprocess.run(
                [sys.executable, script, tmp_in, tmp_out],
                stderr=subprocess.PIPE, text=True, timeout=1200
            )
            if result.returncode != 0:
                print(f'WARNING: RF subprocess failed (exit {result.returncode})'
                      f' — falling back to XGB-only ensemble.')
                if result.stderr:
                    print(f'  stderr: {result.stderr[:300]}')
                return None, None
            try:
                y_rf = np.load(tmp_out).astype(int)
                p_rf = np.load(tmp_proba)
            except FileNotFoundError as e:
                print(f'WARNING: RF output file missing ({e.filename}) '
                      f'— falling back to XGB-only ensemble.')
                return None, None
            return p_rf, y_rf
        finally:
            for p in (tmp_in, tmp_out, tmp_proba):
                if os.path.exists(p):
                    os.unlink(p)

    # ── Output formatter ──────────────────────────────────────────────────
    def build_output(self, meta_df, preds, true_labels=None,
                     lab_attack=None, lab_proba=None):
        out = meta_df.copy()
        out['predicted_class'] = self.le.inverse_transform(preds['y_ensemble'])
        out['confidence']      = preds['confidence'].round(4)
        out['is_anomaly_ae']   = preds['is_anomaly']
        out['recon_mse']       = preds['recon_mse'].round(6)
        alert = (preds['y_ensemble'] != 0) | preds['is_anomaly'].astype(bool)
        if lab_attack is not None:
            # Lab-native detector is the trustworthy attack signal on live lab
            # traffic; OR it into the alert and expose it as its own column.
            out['lab_attack']       = lab_attack.astype(int)
            out['lab_attack_proba'] = np.round(lab_proba, 4)
            alert = alert | (lab_attack == 1)
        out['alert']           = alert.astype(int)
        out['xgb_class']       = self.le.inverse_transform(preds['y_xgb'])
        out['mlp_class']       = self.le.inverse_transform(preds['y_mlp'])
        if preds['y_rf'] is not None:
            out['rf_class'] = self.le.inverse_transform(preds['y_rf'])
        if true_labels is not None:
            out['true_label'] = true_labels
        return out


# ── Main ───────────────────────────────────────────────────────────────────

def run(args):
    print('Loading ML-IDS pipeline...')
    pipeline = IDSPipeline()

    # Collect CSV files
    if os.path.isdir(args.input):
        csv_files = sorted(glob.glob(os.path.join(args.input, '*.csv')))
    else:
        csv_files = [args.input]

    if not csv_files:
        print(f'ERROR: No CSV files found at {args.input}')
        sys.exit(1)

    all_results = []

    for csv_path in csv_files:
        print(f'Processing: {csv_path}')
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f'  Rows: {len(df):,}   Columns: {len(df.columns)}')

        has_label   = 'Label' in df.columns
        true_labels = df['Label'].str.strip().values if has_label else None

        t0 = time.perf_counter()

        X_hybrid, meta_df = pipeline.preprocess(df)
        preds             = pipeline.predict(X_hybrid, use_rf=not args.skip_rf)
        lab_pred, lab_pr  = pipeline.predict_lab(df)
        result_df         = pipeline.build_output(meta_df, preds, true_labels,
                                                  lab_attack=lab_pred, lab_proba=lab_pr)
        if lab_pred is not None:
            print(f'  Lab det. : {int(lab_pred.sum()):,} / {len(df):,} flagged attack')

        elapsed = time.perf_counter() - t0
        fps     = len(df) / elapsed

        if has_label:
            acc          = (result_df['predicted_class'] == result_df['true_label']).mean()
            n_attacks    = (result_df['true_label'] != 'Benign').sum()
            caught       = ((result_df['true_label'] != 'Benign') & (result_df['alert'] == 1)).sum()
            print(f'  Accuracy : {acc:.4f}')
            print(f'  Attacks  : {caught}/{n_attacks} caught  '
                  f'({caught/n_attacks*100:.1f}% recall)' if n_attacks else '')

        print(f'  Alerts   : {result_df["alert"].sum():,} / {len(result_df):,} flows')
        print(f'  Time     : {elapsed:.1f}s  ({fps:.0f} flows/sec)\n')
        all_results.append(result_df)

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_csv(args.output, index=False)

    # Summary
    n_total   = len(combined)
    n_alerts  = int(combined['alert'].sum())
    n_attack  = int((combined['predicted_class'] != 'Benign').sum())
    n_ae_only = int(((combined['is_anomaly_ae'] == 1) &
                     (combined['predicted_class'] == 'Benign')).sum())

    alert_pct = n_alerts / n_total * 100 if n_total else 0.0
    print('=' * 60)
    print(f'  TOTAL FLOWS     : {n_total:,}')
    print(f'  ALERTS RAISED   : {n_alerts:,}  ({alert_pct:.2f}%)')
    print(f'    Classified as attack : {n_attack:,}')
    print(f'    AE-only anomalies    : {n_ae_only:,}')
    if n_attack > 0:
        print(f'\n  Attack class breakdown:')
        breakdown = (combined[combined['predicted_class'] != 'Benign']
                     ['predicted_class'].value_counts())
        for cls, cnt in breakdown.items():
            print(f'    {cls:<30} {cnt:>8,}')
    print(f'\n  Output → {args.output}')
    print('=' * 60)

    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ML-IDS: score CICFlowMeter CSV files with trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input',   required=True,
                        help='CICFlowMeter CSV file or directory of CSVs')
    parser.add_argument('--output',  default='alerts.csv',
                        help='Output CSV path  (default: alerts.csv)')
    parser.add_argument('--skip-rf', action='store_true',
                        help='Skip RF model to save ~11 GB RAM; uses XGB+MLP ensemble only')
    args = parser.parse_args()
    run(args)
