"""Settings for the recognition engine.

Every field can be set with an environment variable named PRAESENS_<FIELD>
(for example PRAESENS_MATCH_THRESHOLD=0.5). The thresholds and cooldown can
also be changed while the server runs through PATCH /faces/config, which is
how we expect to tune them during testing. If PRAESENS_CONFIG_FILE is set,
live changes are written there and loaded again on the next start.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Fields that are safe to change while the engine is running.
TUNABLE_FIELDS = {
    "match_threshold",
    "review_threshold",
    "cooldown_seconds",
    "confirm_frames",
    "min_det_score",
    "min_face_px",
    "process_fps",
}


@dataclass
class Settings:
    # --- Matching -----------------------------------------------------------
    # Cosine similarity between a live face and an enrolled face (range -1..1;
    # same person with buffalo_l is usually well above 0.4, strangers usually
    # below 0.25). These are starting points; tune them with tools/tune_threshold.py.
    match_threshold: float = 0.45
    # Scores between review_threshold and match_threshold are reported as
    # "review" (a possible match the UI can flag) instead of a confident match.
    # Set it equal to match_threshold to turn the review band off.
    review_threshold: float = 0.35
    # A guest triggers at most one notification per cooldown window.
    cooldown_seconds: float = 300.0
    # Number of consecutive processed frames a guest must match before a
    # notification fires. 1 = fire on the first match; 2-3 cuts one-frame flukes.
    confirm_frames: int = 2

    # --- Detection quality --------------------------------------------------
    # Faces below this detector confidence are ignored (live) or rejected (enroll).
    min_det_score: float = 0.6
    # Faces whose shorter bounding-box side is smaller than this many pixels are
    # ignored; tiny faces produce unreliable embeddings.
    min_face_px: int = 40

    # --- Model --------------------------------------------------------------
    model_name: str = "buffalo_l"
    model_root: str = "~/.insightface"
    # -1 = CPU, 0 = first GPU (needs onnxruntime-gpu).
    ctx_id: int = 0
    det_size: int = 640

    # --- Camera -------------------------------------------------------------
    # Webcam index ("0") or a video file / RTSP URL.
    camera_source: str = "0"
    # How many frames per second to run recognition on. The camera is read at
    # full rate; extra frames are only used for the preview stream.
    process_fps: float = 5.0
    # Start the camera when the engine starts.
    camera_autostart: bool = False

    # --- Main API --------------------------------------------------------
    # Base URL of the Praesens API (api/praesens_api.py). On startup and on
    # POST /faces/sync, guests are loaded from GET {api_url}users/.
    api_url: str = "http://localhost:8000/api/v1/"
    # Load guests from the API when the service starts.
    sync_on_startup: bool = True
    # Relative photo_path values from the API are resolved against this folder.
    photo_root: str = "."
    # Comma-separated origins allowed to call this service from a browser
    # (the Svelte dashboard), e.g. "http://localhost:5173". "*" allows all.
    cors_origins: str = "*"

    # --- Files --------------------------------------------------------------
    # Optional JSON file that live config changes are saved to.
    config_file: str = ""
    # Optional CSV that every processed face score is appended to (for tuning).
    score_log_path: str = ""

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        s = cls()
        for f in fields(cls):
            raw = os.environ.get(f"PRAESENS_{f.name.upper()}")
            if raw is not None:
                setattr(s, f.name, _coerce(f.type, raw))
        for k, v in overrides.items():
            setattr(s, k, v)
        if s.config_file and Path(s.config_file).exists():
            saved = json.loads(Path(s.config_file).read_text())
            s.update({k: v for k, v in saved.items() if k in TUNABLE_FIELDS})
        s.validate()
        return s

    def update(self, changes: dict[str, Any]) -> None:
        """Apply live changes (only TUNABLE_FIELDS) and validate them."""
        unknown = set(changes) - TUNABLE_FIELDS
        if unknown:
            raise ValueError(f"not tunable at runtime: {sorted(unknown)}")
        old = asdict(self)
        for k, v in changes.items():
            setattr(self, k, v)
        try:
            self.validate()
        except ValueError:
            for k in changes:
                setattr(self, k, old[k])
            raise

    def save(self) -> None:
        if self.config_file:
            data = {k: getattr(self, k) for k in sorted(TUNABLE_FIELDS)}
            Path(self.config_file).write_text(json.dumps(data, indent=2))

    def validate(self) -> None:
        if not -1.0 <= self.review_threshold <= self.match_threshold <= 1.0:
            raise ValueError("need -1 <= review_threshold <= match_threshold <= 1")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        if self.process_fps <= 0:
            raise ValueError("process_fps must be > 0")

    def tunable(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in sorted(TUNABLE_FIELDS)}


def _coerce(type_name: Any, raw: str) -> Any:
    t = type_name if isinstance(type_name, str) else type_name.__name__
    if t == "bool":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if t == "int":
        return int(raw)
    if t == "float":
        return float(raw)
    return raw
