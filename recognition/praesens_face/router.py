"""FastAPI router exposing the engine. Mount it in the main app with:

    app.include_router(build_router(engine))

All paths are under /faces. Guest ids are strings so they work with either an
integer or a UUID primary key in Postgres; the API layer owns guest metadata,
this router only knows ids and faces.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import EnrollmentError, FaceEngine


class ConfigPatch(BaseModel):
    match_threshold: float | None = None
    review_threshold: float | None = None
    cooldown_seconds: float | None = None
    confirm_frames: int | None = None
    min_det_score: float | None = None
    min_face_px: int | None = None
    process_fps: float | None = None
    persist: bool = True


def build_router(engine: FaceEngine, prefix: str = "/faces") -> APIRouter:
    r = APIRouter(prefix=prefix, tags=["faces"])

    # Plain `def` endpoints run in FastAPI's thread pool, so model inference
    # doesn't block the event loop.

    @r.post("/guests/{guest_id}/enroll")
    def enroll(guest_id: str, photos: list[UploadFile] = File(...)) -> dict[str, Any]:
        """Enroll (or re-enroll) a guest. Each photo must contain exactly one face."""
        data = [p.file.read() for p in photos]
        try:
            n = engine.enroll(guest_id, data)
        except EnrollmentError as e:
            raise HTTPException(422, str(e))
        return {"guest_id": guest_id, "vectors": n}

    @r.delete("/guests/{guest_id}")
    def remove(guest_id: str) -> dict[str, Any]:
        n = engine.remove(guest_id)
        if n == 0:
            raise HTTPException(404, "guest not enrolled")
        return {"guest_id": guest_id, "removed_vectors": n,
                "gallery_fingerprint": engine.gallery.fingerprint()}

    @r.get("/guests")
    def guests() -> dict[str, int]:
        """Enrolled guest ids and how many photos each has."""
        return engine.gallery.guests()

    @r.post("/identify")
    def identify(photo: UploadFile = File(...)) -> list[dict[str, Any]]:
        """Match every face in an uploaded image. Doesn't fire notifications;
        meant for testing and threshold tuning without the camera."""
        try:
            results = engine.identify(photo.file.read())
        except ValueError as e:
            raise HTTPException(422, str(e))
        return [vars(x) for x in results]

    @r.get("/config")
    def get_config() -> dict[str, Any]:
        return engine.settings.tunable()

    @r.patch("/config")
    def patch_config(body: ConfigPatch) -> dict[str, Any]:
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
    def status() -> dict[str, Any]:
        return engine.status()

    @r.post("/camera/start")
    def camera_start() -> dict[str, Any]:
        engine.start_camera()
        return engine.camera.stats()

    @r.post("/camera/stop")
    def camera_stop() -> dict[str, Any]:
        engine.stop_camera()
        return {"running": False}

    @r.get("/stream")
    async def stream() -> StreamingResponse:
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
        """Pushes {"type": "guest_detected", "guest_id": ..., "status": ...,
        "score": ..., "bbox": [...], "timestamp": ...} for each notification."""
        await ws.accept()
        q = engine.subscribe()
        try:
            while True:
                await ws.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            engine.unsubscribe(q)

    return r
