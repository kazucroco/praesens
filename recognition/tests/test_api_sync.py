"""ApiClient against a stand-in for the main API's GET /api/v1/users/."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praesens_face.api_sync import ApiClient  # noqa: E402


@pytest.fixture
def api(tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "ada.jpg").write_bytes(b"ada-bytes")
    users = {
        1: {"first_name": "Henry", "last_name": "Silva", "bio": "", "image_path": "none"},
        2: {"first_name": "Ada", "last_name": "Guest", "bio": "", "image_path": "photos/ada.jpg"},
        3: {"first_name": "Bo", "last_name": "Gone", "bio": "", "image_path": "photos/missing.jpg"},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/v1/users/":
                body = users
            elif self.path.startswith("/api/v1/users/"):
                body = {"id": int(self.path.rsplit("/", 1)[1]), **users[int(self.path.rsplit("/", 1)[1])]}
            else:
                self.send_error(404)
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield ApiClient(f"http://127.0.0.1:{srv.server_port}/api/v1", photo_root=str(tmp_path))
    srv.shutdown()


def test_guests_with_photos(api):
    guests, errors = api.guests_with_photos()
    assert guests == [("2", [b"ada-bytes"])]
    assert set(errors) == {"1", "3"}


def test_single_guest(api):
    assert list(api.iter_guest("2")) == [b"ada-bytes"]
    with pytest.raises(FileNotFoundError):
        list(api.iter_guest("1"))
