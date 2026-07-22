# ML-IDS Live Monitor

A FastAPI + WebSocket dashboard that scores CICFlowMeter flows with the trained
models and shows attacks in real time. Non-benign flows (or autoencoder
anomalies) light up a red banner, get added to the live table with a severity
badge, and are counted per attack type.

Scoring uses **XGBoost (classifier) + Autoencoder (anomaly detector)** — the
`--skip-rf` path from `notebooks/inference.py`. Random Forest is intentionally
not loaded (it needs ~11 GB and ~6 min/run — unusable for live scoring).

```
web/
  app.py               FastAPI server (replay + folder-watch, WebSocket)
  pipeline.py          wraps notebooks/inference.py -> per-flow events
  static/index.html    dashboard (no external dependencies)
  make_demo_replay.py  builds an attack-rich CSV for replay testing
```

## Install & run

```bash
cd web
pip install -r requirements.txt          # ML stack (torch/xgboost/…) already installed
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## 1. Test it now — replay mode (default)

Replays a CSV as if it were live traffic. A ready-made attack-rich file was
generated at `data/live/demo_replay.csv` (Benign + FTP/SSH brute-force):

```bash
uvicorn app:app --port 8000
```

Faster/slower feed:

```bash
REPLAY_INTERVAL=0.3 REPLAY_BATCH=40 uvicorn app:app --port 8000
```

Replay any file you have (e.g. a full day):

```bash
REPLAY_FILE=../data/raw/02-16-2018.csv uvicorn app:app --port 8000
```

Rebuild the demo file from a different day:

```bash
python make_demo_replay.py ../data/raw/02-16-2018.csv 4000
```

## 2. Real attacks — watch mode

Point the server at a folder and have CICFlowMeter write live flows into it:

```bash
mkdir -p ../data/live
SOURCE=watch WATCH_DIR=../data/live uvicorn app:app --host 0.0.0.0 --port 8000
```

The server polls `WATCH_DIR` and scores **new rows** as they are appended.

### Capturing the flows

> **Schema must match.** The models were trained on the **CSE-CIC-IDS-2018**
> 80-column format (Dst Port, Protocol, Timestamp, + 77 flow features). Use the
> **official CICFlowMeter-V4** (the CIC Java tool) so column names line up. The
> Python `cicflowmeter` pip package uses different column names and would need a
> mapping layer first. `pipeline.py` aligns columns by name and validates the
> feature count, so a mismatch fails loudly instead of scoring garbage.

Typical live-capture flow (on the monitoring host / span port):

1. Start CICFlowMeter-V4 in live mode on your capture interface and set its
   output CSV to `data/live/` (or capture to a `.pcap` during the attack, then
   convert: CICFlowMeter reads the pcap and writes the flow CSV into
   `data/live/`).
2. Run the server in `watch` mode (above).
3. Launch your attack against the target (e.g. `nmap`, `hping3` flood,
   `hydra`/FTP-SSH brute force, `slowloris`). Within a couple of seconds the
   flows appear and attacks turn the banner red.

## What a flow shows

| Field | Meaning |
|-------|---------|
| Prediction | XGBoost calibrated class (Benign or an attack type) |
| Conf. | max class probability |
| Severity | CRITICAL / HIGH / MEDIUM / LOW / NONE (by class; LOW = AE-only anomaly) |
| AE | ⚠︎ = autoencoder reconstruction error over threshold (possible novel attack) |
| Ground truth | only when the CSV has a `Label` column (replay/testing) — ✓/✗ vs prediction |

A flow raises an **alert** if the classifier predicts non-Benign **or** the AE
flags it as anomalous — so unseen attack shapes still surface.

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `SOURCE` | `replay` | `replay` or `watch` |
| `REPLAY_FILE` | `data/live/demo_replay.csv` → falls back to `data/raw/02-14-2018.csv` | file to replay |
| `REPLAY_BATCH` | `25` | rows scored per tick |
| `REPLAY_INTERVAL` | `1.0` | seconds between ticks |
| `REPLAY_LOOP` | `1` | restart file when it ends |
| `WATCH_DIR` | `data/live` | folder polled in watch mode |
| `WATCH_INTERVAL` | `2.0` | seconds between folder polls |
| `SKIP_LAB` | `0` | `1` = score with the 2018 XGB + AE only (skip the lab-native detector). Use for the in-distribution 2018 demo so the lab detector doesn't misfire on 2018 benign. |

## Notes / limits

- Scoring is CPU/MPS work run in a thread pool so the event loop stays
  responsive; comfortable at hundreds of flows/sec on a laptop.
- Watch mode tracks per-file row counts in memory — if CICFlowMeter rewrites a
  file from the top, restart the server. New files and appended rows are handled.
- The dashboard keeps the last ~120 flows in the table; counters are cumulative.

## Run / find / kill — cheat-sheet

**Python:** on the HPC use `/application/miniconda/25.5.1/bin/python`; on the x86 box
`conda activate base` then `python`. Examples below use `$PY`.

```bash
cd ~/ML-IDS/web
PY=/application/miniconda/25.5.1/bin/python

# Clean 2018 demo — SSH/FTP brute-force (models on their home distribution)
SKIP_LAB=1 $PY -m uvicorn app:app --host 0.0.0.0 --port 8000

# Lab demo — SYN-Flood/PortScan/DoS via the scale-invariant lab detector
REPLAY_FILE=$HOME/ML-IDS/data/live/lab_demo_replay.csv $PY -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Stop one (Ctrl-C) before starting the other; **hard-refresh** the browser between
runs (Safari `Cmd+Option+R`, Chrome `Cmd+Shift+R`) — the page keeps client-side state.

**Find the server / port 8000:**
```bash
pgrep -af 'uvicorn|app:app'      # all dashboard servers
lsof -i :8000                    # what's listening (or: ss -ltnp | grep :8000)
```

**Kill it / free the port:**
```bash
kill <PID>                       # or: kill -9 <PID>
pkill -f 'uvicorn app:app'       # kill all dashboards
fuser -k 8000/tcp                # force-free the port
lsof -i :8000 || echo "port free"
```
Port stuck? Just run on another port: add `--port 8010`.

**Opening the page (gotchas):**
- `localhost:8000` in your laptop browser = your **laptop**, not the cluster — kill any
  stale local server (`pkill -f 'uvicorn app:app'`) or it shadows the real one.
- **VS Code Remote:** PORTS panel → Forward a Port → `8000` → click 🌐 (don't hand-type localhost).
- **OnDemand Desktop:** run the server there, open `localhost:8000` in that desktop's Firefox (same node).
- **SSH tunnel:** `ssh -L 8000:<node-running-server>:8000 <user>@<login-host>` — target the
  **compute** node's hostname if that's where the server runs.
- Verify the tunnel hits the right server: `curl -s localhost:8000/api/state | head -c 200`.

**Which dataset am I seeing?** 2018 → `14/02/2018`, SSH/FTP, no `192.168.64.2`.
Lab → `18/07/2026`, `Attack (lab-detected)`, lab IPs.
