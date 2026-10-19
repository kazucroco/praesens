"""Camera worker: the one process that opens the webcam.

Only one process can hold the webcam (requirements doc, constraint 4.2), so
the recognition side owns it. One thread reads frames at full rate, a second
runs recognition on the newest frame a few times per second, and the latest
annotated frame is kept as a JPEG, which is served as an MJPEG stream
(GET /faces/stream) so the dashboard can show the live view without opening
the camera itself.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from .engine import FaceEngine, FaceResult

log = logging.getLogger("praesens_face.camera")

COLORS = {"match": (0, 200, 0), "review": (0, 200, 255), "unknown": (160, 160, 160)}


class CameraWorker:
    def __init__(self, engine: "FaceEngine"):
        self.engine = engine
        src = engine.settings.camera_source
        self.source: int | str = int(src) if src.isdigit() else src
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._jpeg_cond = threading.Condition()
        self._frame = None  # newest raw frame, shared with the recognition thread
        self._frame_lock = threading.Lock()
        self._last_results: list["FaceResult"] = []
        self._frames_read = 0
        self._frames_processed = 0
        self._last_latency_ms = 0.0
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="praesens-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def latest_jpeg(self, timeout: float = 1.0, after: bytes | None = None) -> bytes | None:
        """Return the newest preview frame, waiting for one newer than `after`."""
        with self._jpeg_cond:
            self._jpeg_cond.wait_for(lambda: self._jpeg is not None and self._jpeg is not after,
                                     timeout=timeout)
            return self._jpeg

    def stats(self) -> dict:
        return {
            "running": self.running,
            "source": str(self.source),
            "frames_read": self._frames_read,
            "frames_processed": self._frames_processed,
            "last_recognition_ms": round(self._last_latency_ms, 1),
            "error": self.error,
        }

    def _run(self) -> None:
        """Reader loop. Always holds the newest frame, so recognition never
        works on a stale frame sitting in the camera's buffer."""
        cap = None
        is_file = isinstance(self.source, str)
        worker = threading.Thread(target=self._recognize_loop, name="praesens-recognition", daemon=True)
        worker.start()
        try:
            while not self._stop.is_set():
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        self.error = f"cannot open camera source {self.source!r}"
                        log.warning(self.error)
                        self._stop.wait(2.0)
                        continue
                    self.error = None
                    file_delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # end of a video file: stop
                        log.info("camera source ended")
                        break
                    cap.release()
                    cap = None
                    continue
                self._frames_read += 1
                with self._frame_lock:
                    self._frame = frame
                self._publish(frame.copy())
                if file_delay:  # play files at their real speed
                    self._stop.wait(file_delay)
        finally:
            self._stop.set()
            worker.join(timeout=5)
            if cap is not None:
                cap.release()

    def _recognize_loop(self) -> None:
        last_seen = None
        while not self._stop.is_set():
            t_start = time.monotonic()
            with self._frame_lock:
                frame = self._frame
            if frame is not None and frame is not last_seen:
                last_seen = frame
                t0 = time.perf_counter()
                try:
                    self._last_results = self.engine.process_frame(frame)
                except Exception:
                    log.exception("recognition failed on frame")
                self._last_latency_ms = (time.perf_counter() - t0) * 1000
                self._frames_processed += 1
            interval = 1.0 / self.engine.settings.process_fps
            self._stop.wait(max(0.005, interval - (time.monotonic() - t_start)))

    def _publish(self, frame) -> None:
        for r in self._last_results:
            x1, y1, x2, y2 = r.bbox
            color = COLORS[r.status]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{r.guest_id} {r.score:.2f}" if r.status != "unknown" else f"? {r.score:.2f}"
            cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._jpeg_cond:
                self._jpeg = buf.tobytes()
                self._jpeg_cond.notify_all()
