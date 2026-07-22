"""
web/app.py — ML-IDS live monitoring dashboard (FastAPI + WebSocket).

Two flow sources (choose with the SOURCE env var):

  SOURCE=replay   Drip-feed rows from a CSV to simulate a live feed. Use this
                  to test the dashboard with no capture setup. (default)
  SOURCE=watch    Poll WATCH_DIR for CICFlowMeter CSVs and score new rows as
                  they are appended — the real-attack deployment path.

Run:
    cd web
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
    # open http://localhost:8000

Env vars:
    SOURCE          replay | watch                       (default: replay)
    REPLAY_FILE     CSV to replay                         (default: data/live/demo_replay.csv,
                                                           falls back to data/raw/02-14-2018.csv)
    REPLAY_BATCH    rows scored per tick                  (default: 25)
    REPLAY_INTERVAL seconds between ticks                 (default: 1.0)
    REPLAY_LOOP     1 = restart file when it ends         (default: 1)
    WATCH_DIR       folder CICFlowMeter writes into       (default: data/live)
    WATCH_INTERVAL  seconds between folder polls          (default: 2.0)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import asyncio
import glob
import io
import json
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeline import WebScorer

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, '..'))
_STATIC   = os.path.join(_HERE, 'static')

SOURCE          = os.environ.get('SOURCE', 'replay')
REPLAY_BATCH    = int(os.environ.get('REPLAY_BATCH', '25'))
REPLAY_INTERVAL = float(os.environ.get('REPLAY_INTERVAL', '1.0'))
REPLAY_LOOP     = os.environ.get('REPLAY_LOOP', '1') == '1'
WATCH_DIR       = os.environ.get('WATCH_DIR', os.path.join(_ROOT, 'data', 'live'))
WATCH_INTERVAL  = float(os.environ.get('WATCH_INTERVAL', '2.0'))


def _default_replay_file() -> str:
    cand = os.environ.get('REPLAY_FILE')
    if cand:
        return cand
    demo = os.path.join(_ROOT, 'data', 'live', 'demo_replay.csv')
    if os.path.exists(demo):
        return demo
    return os.path.join(_ROOT, 'data', 'raw', '02-14-2018.csv')


class Hub:
    """Holds the scorer, aggregate stats, recent flows, and WS clients."""

    def __init__(self):
        self.scorer   = None                 # loaded on startup (slow-ish)
        self.clients  = set()
        self.recent   = deque(maxlen=200)     # most-recent flow events
        self.by_class = Counter()             # attack class -> count
        self.total_flows  = 0
        self.total_alerts = 0
        self.last_attack  = None
        self.started_at   = time.time()
        self.source       = SOURCE
        self.running      = False

    # ── stats ────────────────────────────────────────────────────────
    def ingest(self, events):
        for e in events:
            self.total_flows += 1
            if e['alert']:
                self.total_alerts += 1
                key = e['predicted_class'] if e['predicted_class'] != 'Benign' \
                    else 'Anomaly (AE)'
                self.by_class[key] += 1
                self.last_attack = {
                    'class':    key,
                    'severity': e['severity'],
                    'ts':       e['ts'],
                    'id':       e['id'],
                    'at':       time.time(),
                }
            self.recent.append(e)

    def snapshot(self) -> dict:
        return {
            'total_flows':  self.total_flows,
            'total_alerts': self.total_alerts,
            'alert_rate':   (self.total_alerts / self.total_flows) if self.total_flows else 0.0,
            'by_class':     dict(self.by_class.most_common()),
            'last_attack':  self.last_attack,
            'uptime_sec':   int(time.time() - self.started_at),
            'source':       self.source,
            'running':      self.running,
            'classes':      self.scorer.classes if self.scorer else [],
        }

    # ── websocket fan-out ────────────────────────────────────────────
    async def broadcast(self, message: dict):
        dead = []
        data = json.dumps(message)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


hub = Hub()


@asynccontextmanager
async def lifespan(_app):
    print('Loading IDS pipeline (XGB + MLP + AE)...')
    hub.scorer = WebScorer()
    print('Pipeline ready.')
    task = asyncio.create_task(watch_loop() if SOURCE == 'watch' else replay_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title='ML-IDS Live Monitor', lifespan=lifespan)


# ── flow sources ─────────────────────────────────────────────────────
async def _score_and_push(df: pd.DataFrame):
    loop = asyncio.get_event_loop()
    events = await loop.run_in_executor(None, hub.scorer.score, df)
    hub.ingest(events)
    await hub.broadcast({'type': 'batch', 'events': events, 'stats': hub.snapshot()})


def _next_chunk(reader):
    try:
        return next(reader)
    except StopIteration:
        return None


async def replay_loop():
    path = _default_replay_file()
    if not os.path.exists(path):
        print(f'[replay] file not found: {path}')
        return
    print(f'[replay] streaming {path}  (batch={REPLAY_BATCH}, interval={REPLAY_INTERVAL}s)')
    loop = asyncio.get_event_loop()
    hub.running = True
    while True:
        reader = pd.read_csv(path, chunksize=REPLAY_BATCH, low_memory=False)
        while True:
            chunk = await loop.run_in_executor(None, _next_chunk, reader)
            if chunk is None:
                break
            try:
                await _score_and_push(chunk)
            except Exception as ex:  # keep the stream alive on a bad chunk
                print(f'[replay] scoring error: {ex}')
            await asyncio.sleep(REPLAY_INTERVAL)
        if not REPLAY_LOOP:
            break
        print('[replay] reached end of file — looping')
    hub.running = False


def _read_appended(path: str, offset: int):
    """Read bytes appended since `offset`, truncated to the last complete line.
    Returns (bytes_or_None, new_offset). Binary mode keeps offsets exact and
    avoids O(n^2) full-file re-reads as the capture grows."""
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read()
    cut = data.rfind(b'\n')
    if cut < 0:
        return None, offset                   # no complete new line yet
    keep = data[:cut + 1]
    return keep, offset + len(keep)


async def watch_loop():
    os.makedirs(WATCH_DIR, exist_ok=True)
    print(f'[watch] polling {WATCH_DIR} every {WATCH_INTERVAL}s for CICFlowMeter CSVs')
    loop = asyncio.get_event_loop()
    state = {}   # path -> {'offset': int (bytes), 'header': bytes}
    hub.running = True
    while True:
        for path in sorted(glob.glob(os.path.join(WATCH_DIR, '*.csv'))):
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            st = state.get(path)
            if st is None:
                # First sight: capture header + score all existing rows.
                try:
                    with open(path, 'rb') as f:
                        header = f.readline()
                    df = await loop.run_in_executor(
                        None, lambda p=path: pd.read_csv(p, low_memory=False))
                except Exception:
                    continue                   # file mid-write; retry next poll
                state[path] = {'offset': size, 'header': header}
                if len(df):
                    try:
                        await _score_and_push(df)
                    except Exception as ex:
                        print(f'[watch] scoring error on {path}: {ex}')
                continue
            if size <= st['offset']:
                continue                       # nothing new (or file truncated)
            keep, new_off = await loop.run_in_executor(
                None, _read_appended, path, st['offset'])
            if not keep:
                continue                       # only a partial line so far
            st['offset'] = new_off
            try:
                df = await loop.run_in_executor(
                    None, lambda b=st['header'] + keep: pd.read_csv(io.BytesIO(b), low_memory=False))
            except Exception as ex:
                print(f'[watch] parse error on {path}: {ex}')
                continue
            if len(df):
                try:
                    await _score_and_push(df)
                except Exception as ex:
                    print(f'[watch] scoring error on {path}: {ex}')
        await asyncio.sleep(WATCH_INTERVAL)


# ── routes ───────────────────────────────────────────────────────────
@app.get('/', response_class=HTMLResponse)
async def index():
    with open(os.path.join(_STATIC, 'index.html')) as f:
        return f.read()


@app.get('/api/state')
async def state():
    return {'stats': hub.snapshot(), 'recent': list(hub.recent)[-100:]}


@app.websocket('/ws')
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    # send current snapshot so a late joiner sees history immediately
    await websocket.send_text(json.dumps({
        'type': 'init',
        'stats': hub.snapshot(),
        'recent': list(hub.recent)[-100:],
    }))
    try:
        while True:
            await websocket.receive_text()   # keepalive; we don't expect input
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)


app.mount('/static', StaticFiles(directory=_STATIC), name='static')
