# Recognition service

The InsightFace part of Praesens. It runs as its own FastAPI app next to the main API (`api/praesens_api.py`): it owns the webcam, keeps enrolled faces in an in-memory FAISS index, loads guests from the main API on startup, and pushes "guest detected" events to the dashboard over a WebSocket. Guest records (names, bios, photo paths) stay in the main API and Postgres; this service only knows guest ids and faces.

```
recognition/
  server.py              The service: startup sync from the main API, CORS, /faces routes
  praesens_face/
    config.py            Settings: thresholds, cooldown, model, camera, API URL (env vars + live PATCH)
    embedder.py          InsightFace buffalo_l wrapper (detection + 512-d ArcFace embedding)
    gallery.py           FAISS IndexIDMap2(IndexFlatIP): add / remove / search / fingerprint
    engine.py            FaceEngine: enroll, identify, process_frame, events
    camera.py            Webcam owner: reader thread + recognition thread + MJPEG preview
    router.py            The /faces routes
    api_sync.py          Reads users and their photos from the main API
  tools/tune_threshold.py  Measure TAR/FAR on your photos and suggest a threshold
  tests/
```

## Running it

From the repo root, with Postgres and the main API already running (see the root README):

```bash
pip install -r recognition/requirements.txt   # on the NVIDIA machines use onnxruntime-gpu instead of onnxruntime
fastapi run recognition/server.py --port 8001
```

The first start downloads buffalo_l (~280 MB) to `~/.insightface`. Then open http://localhost:8001/docs. Without a GPU, set `PRAESENS_CTX_ID=-1`.

Run the tests with `pytest recognition/tests`.

**Run it with one worker** (the default for `fastapi run`). Each worker would load its own model, hold its own gallery and fight over the webcam.

## How it talks to the main API

```
dashboard ──REST──▶ main API :8000 ──▶ Postgres
    │                    ▲
    │                    │ GET /api/v1/users/ (startup, /faces/sync)
    │                    │
    └──WebSocket + MJPEG─┴── recognition :8001 ◀── webcam
```

- **Startup:** the service calls `GET {PRAESENS_API_URL}users/`, reads each user's `image_path` (a local file path, resolved against `PRAESENS_PHOTO_ROOT` if relative, or an http URL) and enrolls them. Users without a readable photo are skipped and logged.
- **New or changed guest:** after the main API saves a guest and their photo, it (or the dashboard) calls `POST /faces/guests/{id}/sync`, and the service re-reads that user from the API.
- **Deleted guest:** call `DELETE /faces/guests/{id}`.
- **Pop-ups:** the dashboard opens `ws://localhost:8001/faces/events`, gets the guest id, and fetches the profile from the main API.

`PRAESENS_API_URL` defaults to `http://localhost:8000/api/v1/`. `PRAESENS_CORS_ORIGINS` (default `*`) controls which browser origins may call this service.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/faces/guests/{guest_id}/enroll` | Multipart `photos` (one or more files). Each must contain exactly one clear face, or nothing is stored and you get 422 with the reason. Re-enrolling replaces the guest's photos. |
| DELETE | `/faces/guests/{guest_id}` | Removes the guest's vectors; returns the new gallery fingerprint. |
| GET | `/faces/guests` | Enrolled ids and photo counts. |
| POST | `/faces/guests/{guest_id}/sync` | Re-enroll one guest from their photo in the main API. |
| POST | `/faces/sync` | Rebuild the whole gallery from the main API. |
| POST | `/faces/identify` | Multipart `photo`; returns every face with `status` (`match` / `review` / `unknown`), best `guest_id` and `score`. No notifications. For testing and tuning. |
| GET / PATCH | `/faces/config` | Read or change thresholds live, e.g. `{"match_threshold": 0.5}`. |
| GET | `/faces/status` | Model, gallery size, fingerprint, camera stats (incl. recognition time per frame). |
| POST | `/faces/camera/start`, `/faces/camera/stop` | Start/stop the webcam worker. |
| GET | `/faces/stream` | MJPEG live view with boxes and scores: `<img src="/faces/stream">`. |
| WS | `/faces/events` | Pushes one JSON message per notification (below). |

WebSocket message:

```json
{"type": "guest_detected", "guest_id": "42", "status": "match",
 "score": 0.71, "bbox": [466, 269, 574, 415], "timestamp": 1791858222.75}
```

The dashboard looks up the guest by id through the main API (`GET /api/v1/users/{id}`) to show the pop-up. `status: "review"` is the low-confidence "possible match" from the proposal; the UI can show it differently or ignore it.

## Thresholds and tuning

Scores are cosine similarity between the live face and the closest enrollment photo. In our end-to-end test a guest scored 0.98 against their own photo and the best stranger scored 0.07, but real webcam frames (angle, lighting, distance, motion blur) score much lower than that, which is why the threshold needs tuning on your own setup.

| Setting | Default | Meaning |
|---|---|---|
| `match_threshold` | 0.45 | At or above: a match, fires a pop-up. |
| `review_threshold` | 0.35 | Between this and match: "review" (possible match). Set equal to `match_threshold` to turn off. |
| `confirm_frames` | 2 | Consecutive processed frames a guest must match before a pop-up. Cuts one-frame flukes. |
| `cooldown_seconds` | 300 | At most one pop-up per guest per window (FR-04 de-duplication). |
| `min_det_score` | 0.6 | Ignore detections the detector is unsure about. |
| `min_face_px` | 40 | Ignore faces smaller than this (too far from the camera). |
| `process_fps` | 5 | Recognition runs per second. The preview runs at full camera rate either way. |

Set any of these with environment variables (`PRAESENS_MATCH_THRESHOLD=0.5`), or live with `PATCH /faces/config`. If `PRAESENS_CONFIG_FILE=praesens_config.json` is set, live changes are saved there and survive restarts. Model and camera settings (`PRAESENS_CTX_ID`, `PRAESENS_DET_SIZE`, `PRAESENS_CAMERA_SOURCE`, `PRAESENS_CAMERA_AUTOSTART`) are env-only.

A tuning loop that fits the test protocol:

1. Set `PRAESENS_SCORE_LOG_PATH=scores.csv` during test sessions. Every processed face is logged with its best guest and score, so you can see the score distributions from real passes.
2. Save webcam frames of each volunteer (and of people who are **not** enrolled) into `dataset/probes/<guest_id>/` and `dataset/probes/unknown/`, with enrollment photos in `dataset/enrolled/<guest_id>/`.
3. Run `python tools/tune_threshold.py dataset --far 0.01`. It prints TAR and FAR across thresholds and suggests one. FAR is only measurable with non-enrolled probes, so get plenty of those (100 unknown probes lets you measure FAR down to 1%).
4. Apply it with `PATCH /faces/config`, rerun passes, repeat.

## Restarts and deletion

The FAISS gallery lives only in RAM and is rebuilt from the stored photos on every start. That makes the stored photo the persistent biometric record, so deleting a guest means deleting their photo and Postgres row in the main API as well as calling `DELETE /faces/guests/{id}` here.

`gallery.fingerprint()` (also in `/faces/status` and the DELETE response) is a SHA-256 over the enrolled ids and vectors, independent of insertion order. For the purge check (H3): enroll A and B, record the fingerprint, enroll C, delete C; the fingerprint must equal the recorded one.

## Performance notes

- On this test container's CPU, recognition on a 6-face frame took about 0.8 s at `det_size=640`. On a GPU with `onnxruntime-gpu` (`PRAESENS_CTX_ID=0`) expect a small fraction of that. On CPU, `PRAESENS_DET_SIZE=320` is noticeably faster but misses small/distant faces.
- The camera reader and recognition run on separate threads, and recognition always takes the newest frame, so slow inference lowers the recognition rate rather than building up lag.
- `/faces/status` reports `last_recognition_ms`, which is the inference part of the NFR-01 latency budget.

## Notes for the main API

- `GET /api/v1/users/` currently inserts a test user ("Henry", no photo) on every call, so each startup sync adds another one. The service skips them because they have no photo, but the table grows.
- Guest ids are treated as strings here, so the integer ids from Postgres work as-is.
- buffalo_l is licensed for non-commercial research use, which is fine for the capstone.
