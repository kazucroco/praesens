"""Shared test helpers for the recognition service.

Nothing here needs a GPU, a webcam, the InsightFace model or a running main
API:

* FakeEmbedder stands in for InsightFace. A test image is a tiny PNG whose
  first pixel value selects which fake faces the embedder "detects" in it.
* fake_api runs a small FastAPI app on a free local port that behaves like the
  main API's /login, /validate and /users endpoints.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

RECOGNITION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RECOGNITION_DIR))

from praesens_face.embedder import DetectedFace  # noqa: E402

DIM = 512
_rng = np.random.default_rng(0)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


ALICE = unit(_rng.normal(size=DIM))
BOB = unit(_rng.normal(size=DIM))
STRANGER = unit(_rng.normal(size=DIM))


def near(base, cos):
    """A unit vector with cosine similarity `cos` to `base`."""
    other = unit(_rng.normal(size=DIM))
    other = unit(other - other.dot(base) * base)
    return unit(cos * base + np.sqrt(1 - cos ** 2) * other)


def face(emb, score=0.9, size=100):
    return DetectedFace((10, 10, 10 + size, 10 + size), score, emb)


class FakeEmbedder:
    dim = DIM

    def __init__(self):
        self.scenes: dict[int, list[DetectedFace]] = {}
        # Faces returned for an image with no registered scene (e.g. camera
        # frames, whose pixels don't survive video compression exactly).
        self.default: list[DetectedFace] = []

    def scene(self, key: int, faces: list[DetectedFace]) -> np.ndarray:
        self.scenes[key] = faces
        img = np.zeros((8, 8, 3), np.uint8)
        img[0, 0, 0] = key
        return img

    def png(self, key, faces) -> bytes:
        return cv2.imencode(".png", self.scene(key, faces))[1].tobytes()

    def detect(self, img):
        return self.scenes.get(int(img[0, 0, 0]), self.default)


# --- fake main API -----------------------------------------------------------

ADMIN_USER = "praesens"
ADMIN_PASSWORD = "changeme"
GOOD_TOKEN = "good-token"


@dataclass
class FakeApiState:
    """What the fake main API serves. Tests edit it directly."""
    users: dict[int, dict[str, Any]] = field(default_factory=dict)
    tokens: set[str] = field(default_factory=lambda: {GOOD_TOKEN})
    validate_calls: int = 0
    login_calls: int = 0
    # Set to an HTTP status to make /validate fail with it (e.g. 500).
    validate_status: int | None = None


class FakeApi:
    def __init__(self, port: int, state: FakeApiState):
        self.port = port
        self.state = state

    @property
    def url(self) -> str:
        """Base URL as in the PRAESENS_API_URL default (with trailing slash)."""
        return f"http://127.0.0.1:{self.port}/api/v1/"


class Login(BaseModel):
    username: str
    password: str


def _build_fake_api(state: FakeApiState):
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    app = FastAPI()
    bearer = HTTPBearer()

    def require_token(c: HTTPAuthorizationCredentials = Depends(bearer)):
        if c.credentials not in state.tokens:
            raise HTTPException(401, "Unauthorized")

    # Same paths and behaviour as database_api/database_api.py.
    @app.post("/api/v1/login")
    def login(body: Login):
        state.login_calls += 1
        if (body.username, body.password) != (ADMIN_USER, ADMIN_PASSWORD):
            raise HTTPException(401, "Unauthorized user")
        return {"access_token": GOOD_TOKEN, "token_type": "bearer"}

    @app.post("/api/v1/validate")
    def validate(c: HTTPAuthorizationCredentials = Depends(bearer)):
        state.validate_calls += 1
        if state.validate_status is not None:
            raise HTTPException(state.validate_status, "boom")
        require_token(c)
        return True

    @app.get("/api/v1/users/", dependencies=[Depends(require_token)])
    def users():
        return state.users

    @app.get("/api/v1/users/{user_id}", dependencies=[Depends(require_token)])
    def user(user_id: int):
        if user_id not in state.users:
            raise HTTPException(404, "User not found")
        return {"id": user_id, **state.users[user_id]}

    @app.get("/photos/{name}")
    def photo(name: str):
        from fastapi.responses import Response
        return Response(b"remote-" + name.encode(), media_type="image/jpeg")

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def _fake_api_server():
    import uvicorn

    state = FakeApiState()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(_build_fake_api(state), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("fake main API did not start")
        time.sleep(0.01)
    yield FakeApi(port, state)
    server.should_exit = True
    t.join(timeout=5)


@pytest.fixture
def fake_api(_fake_api_server):
    """The fake main API, with its state reset for each test."""
    fresh = FakeApiState()
    for k, v in vars(fresh).items():
        setattr(_fake_api_server.state, k, v)
    return _fake_api_server


@pytest.fixture
def dead_api_url() -> str:
    """An API URL nothing is listening on."""
    return f"http://127.0.0.1:{free_port()}/api/v1/"


AUTH = {"Authorization": f"Bearer {GOOD_TOKEN}"}
