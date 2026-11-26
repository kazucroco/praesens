"""ApiClient against the fake main API in conftest.py."""

import pytest
from conftest import GOOD_TOKEN

from praesens_face.api_sync import ApiClient


@pytest.fixture
def api(fake_api, tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "ada.jpg").write_bytes(b"ada-bytes")
    fake_api.state.users.update({
        1: {"first_name": "Henry", "last_name": "Silva", "bio": "", "image_path": "none"},
        2: {"first_name": "Ada", "last_name": "Guest", "bio": "", "image_path": "photos/ada.jpg"},
        3: {"first_name": "Bo", "last_name": "Gone", "bio": "", "image_path": "photos/missing.jpg"},
    })
    return ApiClient(fake_api.url, photo_root=str(tmp_path))


def test_logs_in_on_creation(api, fake_api):
    assert fake_api.state.login_calls == 1
    assert api.headers["Authorization"] == f"Bearer {GOOD_TOKEN}"


def test_url_without_trailing_slash(fake_api):
    api = ApiClient(fake_api.url.rstrip("/"))
    assert api.api_url == fake_api.url
    assert api.users() == {}


def test_guests_with_photos(api):
    guests, errors = api.guests_with_photos()
    assert guests == [("2", [b"ada-bytes"])]
    assert set(errors) == {"1", "3"}
    assert "no photo" in errors["1"] and "photo not readable" in errors["3"]


def test_single_guest(api):
    assert list(api.iter_guest("2")) == [b"ada-bytes"]
    with pytest.raises(FileNotFoundError):
        list(api.iter_guest("1"))


def test_unknown_guest_is_an_os_error(api):
    # server.py turns OSError into a 502, so a 404 from the API must be one.
    with pytest.raises(OSError):
        list(api.iter_guest("99"))


def test_absolute_photo_path(api, tmp_path):
    p = tmp_path / "abs.jpg"
    p.write_bytes(b"abs")
    assert api.read_photo(str(p)) == b"abs"


@pytest.mark.parametrize("path", [None, "", "none", "None"])
def test_no_photo(api, path):
    with pytest.raises(FileNotFoundError, match="no photo"):
        api.read_photo(path)


def test_photo_from_url(api, fake_api):
    url = f"http://127.0.0.1:{fake_api.port}/photos/ada.jpg"
    assert api.read_photo(url) == b"remote-ada.jpg"


def test_expired_token_is_an_os_error(api, fake_api):
    fake_api.state.tokens.clear()
    with pytest.raises(OSError):
        api.users()


def test_main_api_down_does_not_raise_on_creation(dead_api_url):
    # The recognition service must be able to start before the main API does
    # (server.py builds an ApiClient at import time).
    api = ApiClient(dead_api_url)
    with pytest.raises(OSError):
        api.users()


def test_logs_in_later_if_the_api_was_down_at_first(fake_api, dead_api_url):
    api = ApiClient(dead_api_url)
    api.api_url = fake_api.url  # the main API "comes up"
    assert api.users() == {}
    assert fake_api.state.login_calls == 1
