"""Fast tests with a fake embedder (no model download needed).

A test image is a tiny PNG whose first pixel value selects which fake faces
the embedder "detects" in it.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praesens_face import EnrollmentError, FaceEngine, Settings, build_router  # noqa: E402
from praesens_face.embedder import DetectedFace  # noqa: E402

DIM = 512
rng = np.random.default_rng(0)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


ALICE = unit(rng.normal(size=DIM))
BOB = unit(rng.normal(size=DIM))
STRANGER = unit(rng.normal(size=DIM))


def near(base, cos):
    """A unit vector with cosine similarity `cos` to `base`."""
    other = unit(rng.normal(size=DIM))
    other = unit(other - other.dot(base) * base)
    return unit(cos * base + np.sqrt(1 - cos ** 2) * other)


def face(emb, score=0.9, size=100):
    return DetectedFace((10, 10, 10 + size, 10 + size), score, emb)


class FakeEmbedder:
    dim = DIM

    def __init__(self):
        self.scenes: dict[int, list[DetectedFace]] = {}

    def scene(self, key: int, faces: list[DetectedFace]) -> np.ndarray:
        self.scenes[key] = faces
        img = np.zeros((8, 8, 3), np.uint8)
        img[0, 0, 0] = key
        return img

    def png(self, key, faces) -> bytes:
        return cv2.imencode(".png", self.scene(key, faces))[1].tobytes()

    def detect(self, img):
        return self.scenes.get(int(img[0, 0, 0]), [])


@pytest.fixture
def fx():
    emb = FakeEmbedder()
    s = Settings(match_threshold=0.5, review_threshold=0.35, confirm_frames=1, cooldown_seconds=60)
    return FaceEngine(s, embedder=emb), emb


def test_enroll_and_match(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.enroll("bob", [emb.scene(2, [face(BOB)])])
    res = eng.identify(emb.scene(3, [face(near(ALICE, 0.7)), face(near(BOB, 0.4)), face(STRANGER)]))
    assert [(r.status, r.guest_id) for r in res][:2] == [("match", "alice"), ("review", "bob")]
    assert res[2].status == "unknown"
    assert res[0].score == pytest.approx(0.7, abs=1e-4)


def test_enrollment_rejects_bad_photos(fx):
    eng, emb = fx
    with pytest.raises(EnrollmentError, match="no clear face"):
        eng.enroll("x", [emb.scene(1, [])])
    with pytest.raises(EnrollmentError, match="exactly one face"):
        eng.enroll("x", [emb.scene(2, [face(ALICE), face(BOB)])])
    with pytest.raises(EnrollmentError, match="no clear face"):
        eng.enroll("x", [emb.scene(3, [face(ALICE, score=0.3)])])
    # One bad photo out of three leaves the gallery untouched.
    with pytest.raises(EnrollmentError, match="photo 2"):
        eng.enroll("x", [emb.scene(4, [face(ALICE)]), emb.scene(5, []), emb.scene(4, [face(ALICE)])])
    assert eng.gallery.size == 0


def test_multi_photo_enrollment_uses_best_photo(fx):
    eng, emb = fx
    a2 = near(ALICE, 0.6)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)]), emb.scene(2, [face(a2)])])
    assert eng.gallery.guests() == {"alice": 2}
    res = eng.identify(emb.scene(3, [face(a2)]))
    assert res[0].guest_id == "alice" and res[0].score == pytest.approx(1.0, abs=1e-4)


def test_threshold_change_is_live(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    probe = emb.scene(2, [face(near(ALICE, 0.45))])
    assert eng.identify(probe)[0].status == "review"
    eng.settings.update({"match_threshold": 0.4})
    assert eng.identify(probe)[0].status == "match"
    with pytest.raises(ValueError):
        eng.settings.update({"review_threshold": 0.9})  # above match_threshold
    assert eng.settings.review_threshold == 0.35  # rolled back


def test_cooldown_and_confirm_frames(fx):
    eng, emb = fx
    eng.settings.update({"confirm_frames": 2})
    events = []
    eng.add_listener(events.append)
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    seen = emb.scene(2, [face(near(ALICE, 0.8))])
    empty = emb.scene(3, [])
    eng.process_frame(seen)
    assert events == []  # one frame isn't enough
    eng.process_frame(empty)  # streak broken
    eng.process_frame(seen)
    assert events == []
    eng.process_frame(seen)
    assert len(events) == 1 and events[0].guest_id == "alice"
    for _ in range(5):
        eng.process_frame(seen)
    assert len(events) == 1  # cooldown
    eng.settings.update({"cooldown_seconds": 0})
    eng.process_frame(seen)
    assert len(events) == 2


def test_remove_and_fingerprint(fx):
    eng, emb = fx
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.enroll("bob", [emb.scene(2, [face(BOB)])])
    before = eng.gallery.fingerprint()
    eng.enroll("carol", [emb.scene(3, [face(STRANGER)])])
    assert eng.gallery.fingerprint() != before
    assert eng.remove("carol") == 1
    assert eng.gallery.fingerprint() == before
    assert eng.identify(emb.scene(4, [face(STRANGER)]))[0].status == "unknown"


def test_rebuild(fx):
    eng, emb = fx
    errors = eng.rebuild([("alice", [emb.scene(1, [face(ALICE)])]), ("bad", [emb.scene(2, [])])])
    assert list(errors) == ["bad"]
    assert eng.gallery.guests() == {"alice": 1}


def test_score_log(fx, tmp_path):
    eng, emb = fx
    eng.settings.score_log_path = str(tmp_path / "scores.csv")
    eng.enroll("alice", [emb.scene(1, [face(ALICE)])])
    eng.process_frame(emb.scene(2, [face(ALICE), face(STRANGER)]))
    lines = (tmp_path / "scores.csv").read_text().splitlines()
    assert lines[0].startswith("timestamp,guest_id,score") and len(lines) == 3


def test_config_file_roundtrip(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    monkeypatch.setenv("PRAESENS_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("PRAESENS_MATCH_THRESHOLD", "0.55")
    s = Settings.from_env()
    assert s.match_threshold == 0.55
    s.update({"match_threshold": 0.6})
    s.save()
    assert Settings.from_env().match_threshold == 0.6


def test_router(fx):
    eng, emb = fx
    @asynccontextmanager
    async def lifespan(app):
        await eng.startup()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(build_router(eng))

    with TestClient(app) as c:
        r = c.post("/faces/guests/42/enroll", files=[("photos", ("a.png", emb.png(1, [face(ALICE)]), "image/png"))])
        assert r.status_code == 200 and r.json() == {"guest_id": "42", "vectors": 1}
        r = c.post("/faces/guests/43/enroll", files=[("photos", ("b.png", emb.png(2, []), "image/png"))])
        assert r.status_code == 422 and "no clear face" in r.json()["detail"]
        assert c.get("/faces/guests").json() == {"42": 1}

        r = c.post("/faces/identify", files={"photo": ("p.png", emb.png(3, [face(near(ALICE, 0.6))]), "image/png")})
        assert r.json()[0]["guest_id"] == "42" and r.json()[0]["status"] == "match"

        r = c.patch("/faces/config", json={"match_threshold": 0.65, "persist": False})
        assert r.json()["match_threshold"] == 0.65
        assert c.patch("/faces/config", json={"model_name": "x"}).json()["match_threshold"] == 0.65  # ignored field
        assert c.patch("/faces/config", json={"review_threshold": 0.9}).status_code == 422

        with c.websocket_connect("/faces/events") as ws:
            eng.process_frame(emb.scene(4, [face(ALICE)]))
            msg = ws.receive_json()
            assert msg["type"] == "guest_detected" and msg["guest_id"] == "42"

        st = c.get("/faces/status").json()
        assert st["guests_enrolled"] == 1 and st["camera"]["running"] is False
        assert c.delete("/faces/guests/42").status_code == 200
        assert c.delete("/faces/guests/42").status_code == 404
