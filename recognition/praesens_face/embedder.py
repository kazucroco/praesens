"""InsightFace wrapper: image in, detected faces with 512-d embeddings out."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    det_score: float
    embedding: np.ndarray  # float32, shape (512,), L2-normalised

    @property
    def size_px(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return int(min(x2 - x1, y2 - y1))


class Embedder(Protocol):
    dim: int

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]: ...


class InsightFaceEmbedder:
    """Runs InsightFace detection + ArcFace recognition (buffalo_l by default).

    Only the detection and recognition models are loaded; age/gender and the
    landmark models are skipped to save time per frame.
    """

    dim = 512

    def __init__(self, model_name: str = "buffalo_l", root: str = "~/.insightface",
                 ctx_id: int = 0, det_size: int = 640):
        from insightface.app import FaceAnalysis  # heavy import, keep it lazy

        providers = ["CPUExecutionProvider"] if ctx_id < 0 else [
            "CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(name=model_name, root=root, providers=providers,
                                 allowed_modules=["detection", "recognition"])
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        # The ONNX sessions are shared by the camera thread and API requests.
        self._lock = threading.Lock()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        with self._lock:
            faces = self._app.get(image_bgr)
        out = []
        for f in faces:
            x1, y1, x2, y2 = (int(round(v)) for v in f.bbox)
            out.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                det_score=float(f.det_score),
                embedding=np.asarray(f.normed_embedding, dtype=np.float32),
            ))
        return out


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/etc bytes into a BGR array (what InsightFace expects)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img
