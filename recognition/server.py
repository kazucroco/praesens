"""Praesens recognition service.

A separate FastAPI app from the main API (api/praesens_api.py). It owns the
webcam, keeps the FAISS gallery in memory, loads guests from the main API on
startup, and pushes "guest detected" events to the dashboard over a WebSocket.

Run from the repo root, after the main API is up on port 8000:

    fastapi run recognition/server.py --port 8001

Settings come from PRAESENS_* environment variables (see praesens_face/config.py).
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import requests, os

from praesens_face import EnrollmentError, FaceEngine, build_router
from praesens_face.api_sync import ApiClient

security = HTTPBearer()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("praesens_face.server")

engine = FaceEngine()
api = ApiClient(engine.settings.api_url, engine.settings.photo_root)

INTERNAL_ADMIN_UN = os.environ.get("INTERNAL_ADMIN_UN", "praesens")
INTERNAL_ADMIN_PW = os.environ.get("INTERNAL_ADMIN_PW", "changeme")

def _get_auth_headers(username: str, password: str):
    payload = { "username": username, "password": password }
    response = requests.post(engine.settings.api_url + "/login/", json=payload, timeout=10)

    data = response.json()
    token = None
    try:
        token = data.get("access_token")
    except:
        pass
    headers = None

    if token != None:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

    return headers

def _test_auth_headers(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    ext_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
    resp = requests.post(engine.settings.api_url + "/validate", headers=ext_headers, timeout=10)

    return resp.json()

headers = _get_auth_headers(INTERNAL_ADMIN_UN, INTERNAL_ADMIN_PW)

def sync_all() -> dict:
    """Rebuild the gallery from every user in the main API."""
    guests, errors = api.guests_with_photos()
    errors.update(engine.rebuild(guests))
    for gid, err in errors.items():
        log.warning("guest %s not enrolled: %s", gid, err)
    return {"enrolled": len(engine.gallery.guests()), "skipped": errors}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.startup()
    if engine.settings.sync_on_startup:
        try:
            result = await asyncio.to_thread(sync_all)
            log.info("loaded %d guests from %s", result["enrolled"], api.api_url)
        except OSError as e:  # API not running yet; POST /faces/sync later
            log.warning("could not reach the main API at %s: %s", api.api_url, e)
    yield
    await engine.shutdown()


app = FastAPI(title="Praesens recognition", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in engine.settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(build_router(engine))


@app.post("/faces/sync")
def sync(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Reload every guest from the main API (e.g. after bulk changes)."""
    _test_auth_headers(credentials)

    try:
        return sync_all()
    except OSError as e:
        raise HTTPException(502, f"main API not reachable: {e}")


@app.post("/faces/guests/{guest_id}/sync")
def sync_one(
    guest_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> dict:
    """Re-enroll one guest from their photo in the main API. The API (or the
    dashboard) calls this after creating a guest or changing their photo."""
    _test_auth_headers(credentials)

    try:
        n = engine.enroll(guest_id, api.iter_guest(guest_id))
    except EnrollmentError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, f"guest {guest_id} has no readable photo: {e}")
    except OSError as e:
        raise HTTPException(502, f"could not load guest {guest_id} from the main API: {e}")
    return {"guest_id": guest_id, "vectors": n}
