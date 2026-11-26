"""The /faces routes, with the fake main API checking bearer tokens."""

import time
from contextlib import asynccontextmanager

import pytest
from conftest import ALICE, AUTH, FakeEmbedder, face, near
from fastapi import FastAPI
from fastapi.testclient import TestClient

from praesens_face import FaceEngine, Settings, build_router

# Every route except the WebSocket needs a bearer token.
PROTECTED = [
    ("get", "/faces/guests"),
    ("delete", "/faces/guests/42"),
    ("post", "/faces/identify"),
    ("get", "/faces/config"),
    ("patch", "/faces/config"),
    ("get", "/faces/status"),
    ("post", "/faces/camera/start"),
    ("post", "/faces/camera/stop"),
    ("get", "/faces/camera/snapshot"),
    ("get", "/faces/stream"),
]


@pytest.fixture
def emb():
    return FakeEmbedder()


@pytest.fixture
def eng(emb, fake_api, tmp_path):
    s = Settings(match_threshold=0.5, review_threshold=0.35, confirm_frames=1, cooldown_seconds=60,
                 camera_autostart=False, api_url=fake_api.url,
                 config_file=str(tmp_path / "praesens_config.json"))
    eng = FaceEngine(s, embedder=emb)
    yield eng
    eng.stop_camera()


@pytest.fixture
def client(eng):
    @asynccontextmanager
    async def lifespan(app):
        await eng.startup()
        yield
        await eng.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.include_router(build_router(eng))
    with TestClient(app) as c:
        yield c


def png_file(emb, key, faces):
    return {"photo": ("p.png", emb.png(key, faces), "image/png")}


# --- authentication ----------------------------------------------------------

@pytest.mark.parametrize("method, path", PROTECTED)
def test_missing_token_is_rejected(client, eng, method, path):
    r = client.request(method, path)
    assert r.status_code in (401, 403)


# A well-formed request for each route, so only the token is wrong.
VALID_BODY = {
    "/faces/identify": {"files": {"photo": ("p.png", b"x", "image/png")}},
    "/faces/config": {"json": {}},
}


@pytest.mark.parametrize("method, path", PROTECTED)
def test_bad_token_is_rejected(client, eng, fake_api, method, path):
    kwargs = VALID_BODY.get(path, {}) if method != "get" else {}
    r = client.request(method, path, headers={"Authorization": "Bearer wrong"}, **kwargs)
    assert r.status_code == 401, r.text
    assert r.headers["www-authenticate"] == "Bearer"
    assert fake_api.state.validate_calls == 1
    assert eng.camera is None  # camera/start did nothing


def test_api_url_without_trailing_slash(client, eng, fake_api):
    # compose.yaml sets PRAESENS_API_URL without the slash; the default has one.
    eng.settings.api_url = fake_api.url.rstrip("/")
    assert client.get("/faces/guests", headers=AUTH).status_code == 200
    assert client.get("/faces/guests", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_main_api_down_fails_closed(client, eng, dead_api_url):
    eng.settings.api_url = dead_api_url
    r = client.get("/faces/guests", headers=AUTH)
    assert r.status_code == 503


def test_main_api_error_fails_closed(client, fake_api):
    fake_api.state.validate_status = 500
    assert client.get("/faces/guests", headers=AUTH).status_code == 502


# --- guests and identify -----------------------------------------------------

def test_guests_and_delete(client, eng, emb):
    eng.enroll("42", [emb.scene(1, [face(ALICE)])])
    assert client.get("/faces/guests", headers=AUTH).json() == {"42": 1}
    r = client.delete("/faces/guests/42", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"guest_id": "42", "removed_vectors": 1,
                        "gallery_fingerprint": eng.gallery.fingerprint()}
    assert client.get("/faces/guests", headers=AUTH).json() == {}
    assert client.delete("/faces/guests/42", headers=AUTH).status_code == 404


def test_identify(client, eng, emb):
    eng.enroll("42", [emb.scene(1, [face(ALICE)])])
    r = client.post("/faces/identify", headers=AUTH, files=png_file(emb, 2, [face(near(ALICE, 0.6))]))
    [res] = r.json()
    assert res["guest_id"] == "42" and res["status"] == "match"
    assert res["score"] == pytest.approx(0.6, abs=1e-3)
    assert res["bbox"] == [10, 10, 110, 110]


def test_identify_does_not_notify(client, eng, emb):
    events = []
    eng.add_listener(events.append)
    eng.enroll("42", [emb.scene(1, [face(ALICE)])])
    client.post("/faces/identify", headers=AUTH, files=png_file(emb, 2, [face(ALICE)]))
    assert events == []


def test_identify_rejects_non_images(client):
    r = client.post("/faces/identify", headers=AUTH,
                    files={"photo": ("p.txt", b"hello", "text/plain")})
    assert r.status_code == 422 and "decode" in r.json()["detail"]


def test_identify_needs_a_photo(client):
    assert client.post("/faces/identify", headers=AUTH).status_code == 422


# --- config ------------------------------------------------------------------

def test_get_config(client, eng):
    assert client.get("/faces/config", headers=AUTH).json() == eng.settings.tunable()


def test_patch_config(client, eng, tmp_path):
    cfg = tmp_path / "praesens_config.json"
    r = client.patch("/faces/config", headers=AUTH, json={"match_threshold": 0.65, "persist": False})
    assert r.json()["match_threshold"] == 0.65 and eng.settings.match_threshold == 0.65
    assert not cfg.exists()

    r = client.patch("/faces/config", headers=AUTH, json={"cooldown_seconds": 5})
    assert r.status_code == 200 and cfg.exists()
    assert '"cooldown_seconds": 5' in cfg.read_text()


def test_patch_config_ignores_unknown_and_rejects_invalid(client, eng):
    r = client.patch("/faces/config", headers=AUTH, json={"model_name": "x", "persist": False})
    assert r.status_code == 200 and eng.settings.model_name == "buffalo_l"
    r = client.patch("/faces/config", headers=AUTH, json={"review_threshold": 0.9, "persist": False})
    assert r.status_code == 422
    assert eng.settings.review_threshold == 0.35
    r = client.patch("/faces/config", headers=AUTH, json={"confirm_frames": "lots"})
    assert r.status_code == 422


# --- status and camera -------------------------------------------------------

def test_status(client, eng, emb):
    eng.enroll("42", [emb.scene(1, [face(ALICE)])])
    st = client.get("/faces/status", headers=AUTH).json()
    assert st["guests_enrolled"] == 1 and st["vectors"] == 1
    assert st["camera"] == {"running": False}
    assert st["model_loaded"] is True


def test_camera_routes_when_stopped(client):
    assert client.get("/faces/camera/snapshot", headers=AUTH).status_code == 409
    assert client.get("/faces/stream", headers=AUTH).status_code == 409
    assert client.post("/faces/camera/stop", headers=AUTH).json() == {"running": False}


def test_camera_start_snapshot_stop(client, eng, emb, tmp_path):
    from test_camera import make_video, wait_for

    eng.settings.camera_source = str(make_video(tmp_path / "clip.avi", frames=200))
    r = client.post("/faces/camera/start", headers=AUTH)
    assert r.status_code == 200 and r.json()["source"] == eng.settings.camera_source
    assert wait_for(lambda: eng.camera.snapshot_jpeg() is not None)

    r = client.get("/faces/camera/snapshot", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store" and r.content.startswith(b"\xff\xd8")

    assert client.get("/faces/status", headers=AUTH).json()["camera"]["running"] is True
    assert client.post("/faces/camera/stop", headers=AUTH).json() == {"running": False}
    assert client.get("/faces/camera/snapshot", headers=AUTH).status_code == 409


# --- WebSocket ---------------------------------------------------------------

def test_events_websocket(client, eng, emb):
    eng.enroll("42", [emb.scene(1, [face(ALICE)])])
    with client.websocket_connect("/faces/events") as ws:
        assert ws.receive_json() == {"type": "presence", "guest_ids": [],
                                     "timestamp": pytest.approx(time.time(), abs=5)}
        eng.process_frame(emb.scene(2, [face(ALICE)]))
        msg = ws.receive_json()
        assert msg["type"] == "guest_detected" and msg["guest_id"] == "42"
        assert msg["status"] == "match" and msg["bbox"] == [10, 10, 110, 110]
        assert ws.receive_json()["guest_ids"] == ["42"]

        eng.stop_camera()  # clears presence
        assert ws.receive_json()["guest_ids"] == []
        assert client.get("/faces/status", headers=AUTH).json()["websocket_clients"] == 1
    assert wait_until_no_ws_clients(eng)


def test_events_reach_every_client(client, eng, emb):
    eng.enroll("42", [emb.scene(1, [face(ALICE)])])
    with client.websocket_connect("/faces/events") as a, client.websocket_connect("/faces/events") as b:
        a.receive_json(), b.receive_json()  # initial presence
        eng.process_frame(emb.scene(2, [face(ALICE)]))
        assert a.receive_json()["guest_id"] == "42"
        assert b.receive_json()["guest_id"] == "42"


def wait_until_no_ws_clients(eng, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not eng._ws_queues:
            return True
        time.sleep(0.02)
    return False
