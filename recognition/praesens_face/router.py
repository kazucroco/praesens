"""FastAPI router exposing the engine. Mount it in the main app with:

    app.include_router(build_router(engine))

All paths are under /faces. Guest ids are strings so they work with either an
integer or a UUID primary key in Postgres; the API layer owns guest metadata,
this router only knows ids and faces.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import requests

from .engine import EnrollmentError, FaceEngine

security = HTTPBearer()

class ConfigPatch(BaseModel):
    match_threshold: float | None = None
    review_threshold: float | None = None
    cooldown_seconds: float | None = None
    presence_seconds: float | None = None
    confirm_frames: int | None = None
    min_det_score: float | None = None
    min_face_px: int | None = None
    process_fps: float | None = None
    persist: bool = True

def build_router(engine: FaceEngine, prefix: str = "/faces") -> APIRouter:
    r = APIRouter(prefix=prefix, tags=["faces"])

    def _test_auth_headers(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        ext_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        resp = requests.post(engine.settings.api_url + "/validate", headers=ext_headers, timeout=10)

        return resp.json()

    # Plain `def` endpoints run in FastAPI's thread pool, so model inference
    # doesn't block the event loop.
    @r.delete("/guests/{guest_id}")
    def remove(
        guest_id: str,
        credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> dict[str, Any]:
        _test_auth_headers(credentials)
        n = engine.remove(guest_id)
        if n == 0:
            raise HTTPException(404, "guest not enrolled")
        return {"guest_id": guest_id, "removed_vectors": n,
                "gallery_fingerprint": engine.gallery.fingerprint()}

    @r.get("/guests")
    def guests(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, int]:
        _test_auth_headers(credentials)
        """Enrolled guest ids and how many photos each has."""
        return engine.gallery.guests()

    @r.post("/identify")
    def identify(
        photo: UploadFile = File(...),
        credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> list[dict[str, Any]]:
        _test_auth_headers(credentials)
        """Match every face in an uploaded image. Doesn't fire notifications;
        meant for testing and threshold tuning without the camera."""
        try:
            results = engine.identify(photo.file.read())
        except ValueError as e:
            raise HTTPException(422, str(e))
        return [vars(x) for x in results]

    @r.get("/config")
    def get_config(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
        _test_auth_headers(credentials)
        return engine.settings.tunable()

    @r.patch("/config")
    def patch_config(
        body: ConfigPatch,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        ) -> dict[str, Any]:
        _test_auth_headers(credentials)
        """Change thresholds/cooldown live, e.g. {"match_threshold": 0.5}."""
        changes = body.model_dump(exclude_none=True)
        persist = changes.pop("persist", True)
        try:
            engine.settings.update(changes)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if persist:
            engine.settings.save()
        return engine.settings.tunable()

    @r.get("/status")
    def get_status(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
        _test_auth_headers(credentials)
        return engine.status()

    @r.post("/camera/start")
    def camera_start(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
        _test_auth_headers(credentials)
        engine.start_camera()
        return engine.camera.stats()

    @r.post("/camera/stop")
    def camera_stop(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
        _test_auth_headers(credentials)
        engine.stop_camera()
        return {"running": False}

    @r.get("/camera/snapshot")
    def camera_snapshot(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Response:
        _test_auth_headers(credentials)
        """The current camera frame as a JPEG, without boxes. Used to take
        enrollment photos while recognition keeps the camera."""
        cam = engine.camera
        jpeg = cam.snapshot_jpeg() if cam and cam.running else None
        if jpeg is None:
            raise HTTPException(409, "camera is not running")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @r.get("/stream")
    async def stream(credentials: HTTPAuthorizationCredentials = Depends(security)) -> StreamingResponse:
        _test_auth_headers(credentials)
        """MJPEG preview with boxes drawn. Use as <img src="/faces/stream">."""
        cam = engine.camera
        if not cam or not cam.running:
            raise HTTPException(409, "camera is not running")

        async def frames():
            last = None
            while cam.running:
                jpeg = await asyncio.to_thread(cam.latest_jpeg, 1.0, last)
                if jpeg is None or jpeg is last:
                    continue
                last = jpeg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @r.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        q = engine.subscribe()
        await ws.send_json(engine.presence())
        try:
            while True:
                await ws.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            engine.unsubscribe(q)

    return r
