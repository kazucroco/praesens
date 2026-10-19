"""FaceEngine: enrollment, matching, notification de-duplication and events.

This is the one object the API talks to. It owns the embedder, the FAISS
gallery and (optionally) the camera worker.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

from .config import Settings
from .embedder import DetectedFace, Embedder, decode_image
from .gallery import Gallery

log = logging.getLogger("praesens_face")

Status = Literal["match", "review", "unknown"]


class EnrollmentError(ValueError):
    """A photo was rejected (no face, several faces, low quality)."""


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]
    det_score: float
    status: Status
    guest_id: str | None  # best candidate, also set for "review"
    score: float  # cosine similarity to that candidate (-1 if gallery empty)


@dataclass
class GuestEvent:
    """Sent to listeners and WebSocket clients when a guest is recognised."""
    type: Literal["guest_detected"]
    guest_id: str
    status: Status  # "match" or "review"
    score: float
    bbox: tuple[int, int, int, int]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Listener = Callable[[GuestEvent], None]


class FaceEngine:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None):
        self.settings = settings or Settings.from_env()
        self._embedder = embedder
        self._embedder_lock = threading.Lock()
        self.gallery = Gallery(dim=embedder.dim if embedder else 512)
        self._listeners: list[Listener] = []
        self._ws_queues: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_notified: dict[str, float] = {}
        self._streak: dict[str, int] = {}
        self._state_lock = threading.Lock()
        self._score_log_lock = threading.Lock()
        self.camera = None  # set by start_camera()

    # --- lifecycle ----------------------------------------------------------
    @property
    def embedder(self) -> Embedder:
        # Loading the model takes a few seconds, so do it on first use (or in
        # startup()) rather than at import time.
        with self._embedder_lock:
            if self._embedder is None:
                from .embedder import InsightFaceEmbedder
                s = self.settings
                self._embedder = InsightFaceEmbedder(s.model_name, s.model_root, s.ctx_id, s.det_size)
            return self._embedder

    async def startup(self) -> None:
        """Call from the FastAPI lifespan. Loads the model and grabs the loop."""
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(lambda: self.embedder)
        if self.settings.camera_autostart:
            self.start_camera()

    async def shutdown(self) -> None:
        self.stop_camera()

    def start_camera(self) -> None:
        from .camera import CameraWorker
        if self.camera and self.camera.running:
            return
        self.camera = CameraWorker(self)
        self.camera.start()

    def stop_camera(self) -> None:
        if self.camera:
            self.camera.stop()

    # --- enrollment ---------------------------------------------------------
    def embed_enrollment_photo(self, image: bytes | np.ndarray) -> np.ndarray:
        """Return the embedding of the single face in an enrollment photo.

        Raises EnrollmentError unless exactly one usable face is found.
        """
        img = decode_image(image) if isinstance(image, (bytes, bytearray)) else image
        faces = self.embedder.detect(img)
        s = self.settings
        usable = [f for f in faces if f.det_score >= s.min_det_score and f.size_px >= s.min_face_px]
        if not usable:
            raise EnrollmentError("no clear face found in photo")
        if len(usable) > 1:
            raise EnrollmentError(f"photo must contain exactly one face, found {len(usable)}")
        return usable[0].embedding

    def enroll(self, guest_id: str, photos: Iterable[bytes | np.ndarray], replace: bool = True) -> int:
        """Enroll a guest from one or more photos. All photos must pass, or
        nothing is changed. Returns the number of vectors stored for the guest."""
        guest_id = str(guest_id)
        embs = []
        for i, p in enumerate(photos):
            try:
                embs.append(self.embed_enrollment_photo(p))
            except (EnrollmentError, ValueError) as e:
                raise EnrollmentError(f"photo {i + 1}: {e}") from e
        if not embs:
            raise EnrollmentError("at least one photo is required")
        mat = np.stack(embs)
        if replace:
            self.gallery.replace(guest_id, mat)
        else:
            self.gallery.add(guest_id, mat)
        self._forget(guest_id)
        return self.gallery.guests()[guest_id]

    def remove(self, guest_id: str) -> int:
        guest_id = str(guest_id)
        n = self.gallery.remove(guest_id)
        self._forget(guest_id)
        return n

    def rebuild(self, guests: Iterable[tuple[str, Iterable[bytes | np.ndarray]]]) -> dict[str, str]:
        """Re-create the gallery from stored photos, e.g. on startup.

        `guests` yields (guest_id, photos). Returns {guest_id: error} for any
        guest that could not be enrolled; the rest are loaded.
        """
        self.gallery.clear()
        errors = {}
        for gid, photos in guests:
            try:
                self.enroll(gid, photos)
            except EnrollmentError as e:
                errors[str(gid)] = str(e)
        return errors

    def _forget(self, guest_id: str) -> None:
        with self._state_lock:
            self._last_notified.pop(guest_id, None)
            self._streak.pop(guest_id, None)

    # --- matching -----------------------------------------------------------
    def identify(self, image: bytes | np.ndarray) -> list[FaceResult]:
        """Detect every face in an image and match it against the gallery."""
        img = decode_image(image) if isinstance(image, (bytes, bytearray)) else image
        s = self.settings
        faces = [f for f in self.embedder.detect(img)
                 if f.det_score >= s.min_det_score and f.size_px >= s.min_face_px]
        return self._match(faces)

    def _match(self, faces: list[DetectedFace]) -> list[FaceResult]:
        if not faces:
            return []
        s = self.settings
        hits = self.gallery.search(np.stack([f.embedding for f in faces]), top_k=1)
        results = []
        for f, h in zip(faces, hits):
            gid, score = (h[0].guest_id, h[0].score) if h else (None, -1.0)
            if score >= s.match_threshold:
                status: Status = "match"
            elif score >= s.review_threshold:
                status = "review"
            else:
                status = "unknown"
            results.append(FaceResult(f.bbox, f.det_score, status, gid, float(score)))
        return results

    def process_frame(self, frame: np.ndarray) -> list[FaceResult]:
        """Identify faces in a live frame and fire notifications. Used by the
        camera worker; can also be called directly with frames from elsewhere."""
        results = self.identify(frame)
        self._log_scores(results)
        now = time.time()
        s = self.settings
        events = []
        with self._state_lock:
            # The best result per guest in this frame (a guest can only be one face).
            seen: dict[str, FaceResult] = {}
            for r in results:
                if r.status != "unknown" and r.guest_id is not None:
                    if r.guest_id not in seen or r.score > seen[r.guest_id].score:
                        seen[r.guest_id] = r
            # Streaks reset for guests not seen in this frame.
            self._streak = {g: self._streak.get(g, 0) + 1 for g in seen}
            for g, r in seen.items():
                if self._streak[g] < s.confirm_frames:
                    continue
                last = self._last_notified.get(g)
                if last is not None and now - last < s.cooldown_seconds:
                    continue
                self._last_notified[g] = now
                events.append(GuestEvent("guest_detected", g, r.status, r.score, r.bbox, now))
        for e in events:
            self._emit(e)
        return results

    # --- events -------------------------------------------------------------
    def add_listener(self, fn: Listener) -> None:
        """Register a callback for every GuestEvent (e.g. to write the event
        log to Postgres). Called from the camera thread, so keep it quick."""
        self._listeners.append(fn)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._ws_queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._ws_queues.discard(q)

    def _emit(self, event: GuestEvent) -> None:
        log.info("guest %s detected (%s, %.3f)", event.guest_id, event.status, event.score)
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:
                log.exception("event listener failed")
        if self._loop is not None:
            payload = event.to_dict()
            for q in list(self._ws_queues):
                self._loop.call_soon_threadsafe(_put_nowait, q, payload)

    def _log_scores(self, results: list[FaceResult]) -> None:
        path = self.settings.score_log_path
        if not path or not results:
            return
        with self._score_log_lock:
            new = not Path(path).exists()
            with open(path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["timestamp", "guest_id", "score", "status", "det_score", "face_px"])
                t = f"{time.time():.3f}"
                for r in results:
                    px = min(r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1])
                    w.writerow([t, r.guest_id or "", f"{r.score:.4f}", r.status, f"{r.det_score:.3f}", px])

    def status(self) -> dict[str, Any]:
        cam = self.camera
        return {
            "model_loaded": self._embedder is not None,
            "guests_enrolled": len(self.gallery.guests()),
            "vectors": self.gallery.size,
            "gallery_fingerprint": self.gallery.fingerprint(),
            "camera": cam.stats() if cam else {"running": False},
            "websocket_clients": len(self._ws_queues),
            "config": self.settings.tunable(),
        }


def _put_nowait(q: asyncio.Queue, item: Any) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:  # a stalled client shouldn't block everyone else
        pass
