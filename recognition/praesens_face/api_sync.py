"""Loads enrolled guests from the main Praesens API.

The API owns guest records in Postgres. Each user has a photo_path; this
module reads those photos so the in-memory FAISS gallery can be rebuilt on
startup, or refreshed for one guest after the API creates or edits them.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import requests
from pathlib import Path
from typing import Any, Iterator

import os

log = logging.getLogger("praesens_face.api_sync")

INTERNAL_ADMIN_UN = os.environ.get("INTERNAL_ADMIN_UN", "praesens")
INTERNAL_ADMIN_PW = os.environ.get("INTERNAL_ADMIN_PW", "changeme")

class ApiClient:
    def __init__(self, api_url: str, photo_root: str = ".", timeout: float = 10.0):
        self.api_url = api_url if api_url.endswith("/") else api_url + "/"
        self.photo_root = Path(photo_root)
        self.timeout = timeout

        self.headers = {}
        # Log in now if the main API is up; otherwise on first use, so the
        # recognition service can start before the main API does.
        try:
            self.get_token(INTERNAL_ADMIN_UN, INTERNAL_ADMIN_PW)
        except requests.RequestException as e:
            log.warning("could not log in to the main API at %s: %s", self.api_url, e)

    def _get_json(self, path: str) -> Any:
        if not self.headers:
            self.get_token(INTERNAL_ADMIN_UN, INTERNAL_ADMIN_PW)
        req = urllib.request.Request(self.api_url + path, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def users(self) -> dict[str, dict[str, Any]]:
        """GET users/ -> {user_id: {first_name, last_name, bio, image_path}}."""
        return {str(k): v for k, v in self._get_json("users/").items()}

    def user(self, user_id: str) -> dict[str, Any]:
        return self._get_json(f"users/{user_id}")

    def read_photo(self, image_path: str | None) -> bytes:
        """Read a user's photo from a local path or an http(s) URL."""
        if not image_path or image_path.lower() == "none":
            raise FileNotFoundError("user has no photo")
        if image_path.startswith(("http://", "https://")):
            # Only send our token to the main API, not to other photo hosts.
            headers = self.headers if image_path.startswith(self.api_url) else {}
            req = urllib.request.Request(image_path, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        p = Path(image_path)
        if not p.is_absolute():
            p = self.photo_root / p
        return p.read_bytes()

    def guests_with_photos(self) -> tuple[list[tuple[str, list[bytes]]], dict[str, str]]:
        """All users as (guest_id, [photo]) pairs, plus {guest_id: error} for
        users whose photo couldn't be read."""
        guests, errors = [], {}
        for uid, u in self.users().items():
            try:
                guests.append((uid, [self.read_photo(u.get("image_path"))]))
            except (OSError, ValueError) as e:
                errors[uid] = f"photo not readable: {e}"
        return guests, errors

    def iter_guest(self, user_id: str) -> Iterator[bytes]:
        yield self.read_photo(self.user(user_id).get("image_path"))

    def get_token(self, username: str, password: str):
        payload = {"username": username, "password": password}

        response = requests.post(self.api_url + "login/", json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()

        token = data.get("access_token")

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
