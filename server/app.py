"""
FastAPI server for padel-scorekeeper.

POST /calibrate   — init CourtMapper + StreamPipeline from 4 corners
POST /frame       — JPEG frame → inference → broadcast to display WS clients
WS   /ws          — display clients
GET  /status      — health check
GET  /history     — game records
POST /history     — save game record
GET  /            — serves webapp/index.html
"""
import asyncio
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from calibrate_court import COURT_PTS_M, CourtMapper
from stream_pipeline import StreamPipeline

app = FastAPI(title='Padel Scorekeeper')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

WEBAPP       = Path(__file__).parent.parent / 'webapp' / 'index.html'
HISTORY_FILE = Path(__file__).parent.parent / 'data' / 'history.json'
HISTORY_FILE.parent.mkdir(exist_ok=True)

_pipeline: Optional[StreamPipeline] = None
_frame_idx: int = 0
_last_result: Optional[dict] = None
_display_clients: list[WebSocket] = []
_executor = ThreadPoolExecutor(max_workers=1)

# Player label assignment: near half = A1/A2, far half = B1/B2
_NEAR = (38, 149, 255)
_FAR  = (60, 220, 60)
_BALL = (255, 240, 60)


def _annotate(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    H_inv = None
    if _pipeline is not None:
        try:
            H_inv = np.linalg.inv(_pipeline.mapper.H)
        except Exception:
            pass

    near_count = far_count = 0
    for p in result.get('players', []):
        color = _NEAR if p['half'] == 'near' else _FAR
        if p['half'] == 'near':
            near_count += 1; label = f"A{near_count}"
        else:
            far_count += 1;  label = f"B{far_count}"
        if H_inv is not None:
            pt = cv2.perspectiveTransform(
                np.array([[[p['x_m'], p['y_m']]]], dtype=np.float32), H_inv)[0][0]
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(out, (px, py), 22, (0, 0, 0), 5)
            cv2.circle(out, (px, py), 22, color, 2)
            cv2.putText(out, label, (px - 14, py + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out, label, (px - 14, py + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ball = result.get('ball')
    if ball:
        bx, by = int(ball['x']), int(ball['y'])
        cv2.circle(out, (bx, by), 10, _BALL, -1)
        cv2.circle(out, (bx, by), 12, (255, 255, 255), 1)

    spd = result.get('speed_kmh')
    if spd:
        txt = f'{spd:.0f} km/h'
        cv2.putText(out, txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4)
        cv2.putText(out, txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, _BALL, 2)

    sc = result.get('score', {})
    na, nb = sc.get('name_a', 'A'), sc.get('name_b', 'B')
    sa, sb = sc.get('sets_a', 0), sc.get('sets_b', 0)
    ga, gb = sc.get('a', 0), sc.get('b', 0)
    pa, pb = sc.get('pts_a', '0'), sc.get('pts_b', '0')
    txt = f"{na} {sa}/{ga}/{pa}  –  {sb}/{gb}/{pb} {nb}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    return out


def _encode_frame(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    scale = min(1.0, 640 / w)
    if scale < 1.0:
        frame = cv2.resize(frame, (640, int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buf).decode()


async def _broadcast(result: dict, raw_frame: Optional[np.ndarray] = None):
    payload = dict(result)
    if raw_frame is not None and _display_clients:
        payload['frame_b64'] = _encode_frame(_annotate(raw_frame, result))
    msg = json.dumps(payload)
    dead = []
    for ws in _display_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _display_clients.remove(ws)


class CalibrateRequest(BaseModel):
    corners: list[list[float]]
    frame_w: int = 854
    frame_h: int = 480
    model: str = 'yolo11n.pt'
    conf: float = 0.25
    court_margin: float = 0.5
    init_score: Optional[str] = None
    first_server: str = 'A'
    name_a: str = 'A'
    name_b: str = 'B'
    fps: float = 10.0
    golden_point: bool = False
    # TrackNetV3 — if not sent, fall back to env vars set by kernel.py
    tracknet_repo: Optional[str] = None
    tracknet_ckpt: Optional[str] = None
    inpaint_ckpt: Optional[str] = None


@app.post('/calibrate')
async def calibrate(req: CalibrateRequest):
    global _pipeline, _frame_idx
    corners = np.float32(req.corners)
    H, _ = cv2.findHomography(corners, COURT_PTS_M)
    if H is None:
        raise HTTPException(400, 'Homography failed — corners likely collinear')
    mapper = CourtMapper(H, corners, (req.frame_w, req.frame_h))
    init = None
    if req.init_score:
        a, b = (int(x) for x in req.init_score.split(','))
        init = {'A': a, 'B': b}
    tn_repo = req.tracknet_repo or os.environ.get('TRACKNET_REPO')
    tn_ckpt = req.tracknet_ckpt or os.environ.get('TRACKNET_CKPT')
    ip_ckpt = req.inpaint_ckpt  or os.environ.get('INPAINT_CKPT')
    if tn_repo:
        print(f'[calibrate] TrackNetV3 enabled — repo={tn_repo}', flush=True)
    else:
        print('[calibrate] TrackNetV3 not configured — ball tracking disabled', flush=True)
    _pipeline = StreamPipeline(
        mapper=mapper, model_path=req.model, conf=req.conf,
        court_margin=req.court_margin, fps=req.fps,
        tracknet_repo=tn_repo, tracknet_ckpt=tn_ckpt, inpaint_ckpt=ip_ckpt,
        initial_score=init, first_server=req.first_server,
        names={'A': req.name_a, 'B': req.name_b},
        golden_point=req.golden_point,
    )
    _frame_idx = 0
    print(f'[calibrate] ready — corners={req.corners}', flush=True)
    return {'status': 'ok'}


@app.post('/frame')
async def process_frame(file: UploadFile = File(...)):
    global _frame_idx, _last_result
    if _pipeline is None:
        raise HTTPException(400, 'Not calibrated')
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, 'Could not decode image')
    idx = _frame_idx
    _frame_idx += 1
    if _executor._work_queue.qsize() > 0:
        return {'status': 'dropped', 'frame': idx}
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _pipeline.process, frame, idx)
    _last_result = result
    await _broadcast(result, frame)
    return result


@app.websocket('/ws')
async def display_ws(websocket: WebSocket):
    await websocket.accept()
    _display_clients.append(websocket)
    if _last_result:
        await websocket.send_text(json.dumps(_last_result))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _display_clients:
            _display_clients.remove(websocket)


@app.get('/status')
async def status():
    return {'calibrated': _pipeline is not None, 'frame': _frame_idx,
            'display_clients': len(_display_clients)}


@app.get('/history')
async def get_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


@app.post('/history')
async def save_game(game: dict):
    history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    history.insert(0, game)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    return {'saved': True, 'total': len(history)}


@app.get('/')
async def root():
    if WEBAPP.exists():
        return FileResponse(WEBAPP, media_type='text/html')
    return JSONResponse({'status': 'webapp not found'})
