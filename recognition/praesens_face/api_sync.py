"""Loads enrolled guests from the main Praesens API.

The API owns guest records in Postgres. Each user has a photo_path; this
module reads those photos so the in-memory FAISS gallery can be rebuilt on
startup, or refreshed for one guest after the API creates or edits them.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("praesens_face.api_sync")


class ApiClient:
    def __init__(self, api_url: str, photo_root: str = ".", timeout: float = 10.0):
        self.api_url = api_url if api_url.endswith("/") else api_url + "/"
        self.photo_root = Path(photo_root)
        self.timeout = timeout

    def _get_json(self, path: str) -> Any:
        with urllib.request.urlopen(self.api_url + path, timeout=self.timeout) as resp:
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
            with urllib.request.urlopen(image_path, timeout=self.timeout) as resp:
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
