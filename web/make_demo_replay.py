"""
Build a small, attack-rich CSV for testing the dashboard in replay mode.

Samples benign + every attack class present in a source raw CSV, interleaves
them so attacks surface quickly on the dashboard, and writes it to
data/live/demo_replay.csv (same 80-column schema the models expect).

Usage:
    python make_demo_replay.py                       # uses 02-14-2018.csv
    python make_demo_replay.py ../data/raw/02-16-2018.csv 4000
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

SRC   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, 'data', 'raw', '02-14-2018.csv')
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
OUT   = os.path.join(_ROOT, 'data', 'live', 'demo_replay.csv')
PER_ATTACK = 400        # cap rows kept per attack class
rng = np.random.default_rng(42)

print(f'Reading {SRC} in chunks...')
benign, attacks = [], []
benign_target = TOTAL
for chunk in pd.read_csv(SRC, chunksize=200_000, low_memory=False):
    chunk = chunk[chunk['Label'] != 'Label']            # drop rogue header rows
    lab = chunk['Label'].astype(str).str.strip()
    for cls, grp in chunk.groupby(lab):
        if cls == 'Benign':
            if sum(len(b) for b in benign) < benign_target:
                benign.append(grp.sample(min(len(grp), 400), random_state=42))
        else:
            attacks.append((cls, grp))
    got_attacks = {c for c, _ in attacks}
    if sum(len(b) for b in benign) >= benign_target and len(got_attacks) >= 1:
        # keep scanning a little to collect more attack variety, then stop
        if len(attacks) > 30:
            break

# cap each attack class
by_cls = {}
for cls, grp in attacks:
    by_cls.setdefault(cls, []).append(grp)
attack_frames = []
for cls, frames in by_cls.items():
    df = pd.concat(frames, ignore_index=True)
    attack_frames.append(df.sample(min(len(df), PER_ATTACK), random_state=42))
    print(f'  attack "{cls}": {min(len(df), PER_ATTACK)} rows')

benign_df = pd.concat(benign, ignore_index=True).head(TOTAL // 2)
print(f'  benign: {len(benign_df)} rows')

demo = pd.concat([benign_df] + attack_frames, ignore_index=True)
demo = demo.sample(frac=1.0, random_state=7).reset_index(drop=True)   # interleave

os.makedirs(os.path.dirname(OUT), exist_ok=True)
demo.to_csv(OUT, index=False)
print(f'\nWrote {len(demo):,} rows -> {OUT}')
print('Label mix:')
print(demo['Label'].astype(str).str.strip().value_counts().to_string())
