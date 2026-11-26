"""server.py: startup sync from the main API and the /faces/sync routes.

server.py builds its engine and API client at import time from PRAESENS_*
environment variables, so each test imports a fresh copy pointed at the fake
main API, with the fake embedder swapped in before startup.
"""

import importlib
import sys

import pytest
from conftest import ALICE, AUTH, BOB, FakeEmbedder, face
from fastapi.testclient import TestClient


@pytest.fixture
def emb():
    return FakeEmbedder()


@pytest.fixture
def photos(tmp_path, emb):
    """Write enrollment photos into a photo root and return a helper."""
    root = tmp_path / "images"
    root.mkdir()

    def write(name, key, faces):
        (root / name).write_bytes(emb.png(key, faces))
        return name

    write.root = root
    return write


@pytest.fixture
def load_server(monkeypatch, photos, emb, fake_api):
    def load(api_url=None, sync_on_startup=True):
        monkeypatch.setenv("PRAESENS_API_URL", api_url or fake_api.url)
        monkeypatch.setenv("PRAESENS_PHOTO_ROOT", str(photos.root))
        monkeypatch.setenv("PRAESENS_CAMERA_AUTOSTART", "0")
        monkeypatch.setenv("PRAESENS_SYNC_ON_STARTUP", "1" if sync_on_startup else "0")
        monkeypatch.setenv("PRAESENS_CONFIG_FILE", "")
        monkeypatch.setenv("PRAESENS_CORS_ORIGINS", "http://dashboard.test, http://other.test")
        sys.modules.pop("server", None)
        server = importlib.import_module("server")
        server.engine._embedder = emb  # no model download
        return server

    yield load
    sys.modules.pop("server", None)


def user(image_path, first="Guest"):
    return {"first_name": first, "last_name": "Test", "bio": "", "image_path": image_path}


def test_startup_loads_guests_from_the_main_api(load_server, fake_api, photos):
    fake_api.state.users.update({
        1: user(photos("alice.png", 1, [face(ALICE)])),
        2: user(photos("bob.png", 2, [face(BOB)])),
        3: user("none"),
        4: user(photos("group.png", 3, [face(ALICE), face(BOB)])),
    })
    server = load_server()
    with TestClient(server.app) as c:
        assert server.engine.gallery.guests() == {"1": 1, "2": 1}
        assert c.get("/faces/guests", headers=AUTH).json() == {"1": 1, "2": 1}


def test_startup_skips_sync_when_turned_off(load_server, fake_api, photos):
    fake_api.state.users[1] = user(photos("alice.png", 1, [face(ALICE)]))
    server = load_server(sync_on_startup=False)
    with TestClient(server.app):
        assert server.engine.gallery.guests() == {}


def test_starts_while_main_api_is_down(load_server, dead_api_url):
    server = load_server(api_url=dead_api_url)
    with TestClient(server.app) as c:
        assert server.engine.gallery.guests() == {}
        # Nothing can be validated either, so protected routes fail closed.
        assert c.post("/faces/sync", headers=AUTH).status_code == 503


def test_sync_all(load_server, fake_api, photos):
    server = load_server(sync_on_startup=False)
    with TestClient(server.app) as c:
        fake_api.state.users.update({
            1: user(photos("alice.png", 1, [face(ALICE)])),
            2: user("none"),
        })
        r = c.post("/faces/sync", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["enrolled"] == 1 and list(body["skipped"]) == ["2"]

        # A guest deleted in the main API disappears on the next full sync.
        del fake_api.state.users[1]
        assert c.post("/faces/sync", headers=AUTH).json()["enrolled"] == 0
        assert server.engine.gallery.guests() == {}


def test_sync_one(load_server, fake_api, photos):
    server = load_server(sync_on_startup=False)
    with TestClient(server.app) as c:
        fake_api.state.users[7] = user(photos("alice.png", 1, [face(ALICE)]))
        r = c.post("/faces/guests/7/sync", headers=AUTH)
        assert r.status_code == 200 and r.json() == {"guest_id": "7", "vectors": 1}

        # New photo of someone else: re-sync replaces the old vector.
        fake_api.state.users[7] = user(photos("bob.png", 2, [face(BOB)]))
        assert c.post("/faces/guests/7/sync", headers=AUTH).json()["vectors"] == 1
        [res] = server.engine.identify(photos.root.joinpath("bob.png").read_bytes())
        assert res.guest_id == "7" and res.status == "match"


@pytest.mark.parametrize("image_path, faces, status, detail", [
    ("none", None, 404, "no readable photo"),
    ("missing.png", None, 404, "no readable photo"),
    ("empty.png", [], 422, "no clear face"),
    ("two.png", [face(ALICE), face(BOB)], 422, "exactly one face"),
])
def test_sync_one_errors(load_server, fake_api, photos, image_path, faces, status, detail):
    if faces is not None:
        photos(image_path, 9, faces)
    fake_api.state.users[5] = user(image_path)
    server = load_server(sync_on_startup=False)
    with TestClient(server.app) as c:
        r = c.post("/faces/guests/5/sync", headers=AUTH)
        assert r.status_code == status and detail in r.json()["detail"]
        assert server.engine.gallery.guests() == {}


def test_sync_one_unknown_guest(load_server):
    server = load_server(sync_on_startup=False)
    with TestClient(server.app) as c:
        assert c.post("/faces/guests/404/sync", headers=AUTH).status_code == 502


@pytest.mark.parametrize("path", ["/faces/sync", "/faces/guests/1/sync"])
def test_sync_routes_need_a_valid_token(load_server, path):
    server = load_server(sync_on_startup=False)
    with TestClient(server.app) as c:
        assert c.post(path).status_code in (401, 403)
        assert c.post(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_cors_origins_come_from_settings(load_server):
    server = load_server(sync_on_startup=False)
    with TestClient(server.app) as c:
        r = c.options("/faces/status", headers={"Origin": "http://dashboard.test",
                                                "Access-Control-Request-Method": "GET"})
        assert r.headers["access-control-allow-origin"] == "http://dashboard.test"
        r = c.options("/faces/status", headers={"Origin": "http://evil.test",
                                                "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in r.headers

