import os
import shutil
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import psycopg

import secrets
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from datetime import datetime, timezone

app = FastAPI()
security = HTTPBearer()

DB_HOST = os.environ.get("DB_HOST", "localhost")

INTERNAL_ADMIN_UN = os.environ.get("INTERNAL_ADMIN_UN", "praesens")
INTERNAL_ADMIN_PW = os.environ.get("INTERNAL_ADMIN_PW", "changeme")

API_VERSION = "/api/v1/"
DB_CONNECTION_PARAMS = os.environ.get(
    "PRAESENS_DB_CONNECTION_PARAMS",
    f"host={DB_HOST} port=5432 dbname=praesens user=praesens password=praesens"
)

IMAGE_PATH = "./images/"

# The face recognition service (recognition/server.py). The API tells it about
# new and deleted guests so they're recognized (or forgotten) right away.
RECOGNITION_URL = os.environ.get("PRAESENS_RECOGNITION_URL", "http://praesens_recognition:8001")
RECOGNITION_TIMEOUT = 30.0

SELF_API_CLIENT_TOKEN = None

# Expected body schema for login endpoint
class User(BaseModel):
    username: str
    password: str

# Expected body schema for log submission endpoint
class UserID(BaseModel):
    user_id: int

# Expected body schema for log pull endpoint
class LogBulkParams(BaseModel):
    limit: int | None = 500
    offset: int | None = 0

@app.get(API_VERSION)
def hello():
    return {"message": "Hello world!"}

# Request a token from the server
@app.post(API_VERSION + "login")
def login(user: User):
    username = user.username
    password = user.password

    if len(username) > 30:
        raise HTTPException(
            status_code=400,
            detail="Username must be less than 30 characters."
        )

    __check_create_admin_table()

    ph = PasswordHasher()

    token = None

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM admins WHERE username = %s
            """, (username,))

            admin = cur.fetchone()

            if admin == None or not ph.verify(admin[2], password):
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized user"
                )

            token = secrets.token_urlsafe(32)
            hashed_token = ph.hash(token)

            cur.execute("""
                UPDATE admins
                SET tokens = array_append(tokens, %s)
                WHERE username = %s
            """, (hashed_token, username))

    return { "access_token": token, "token_type": "bearer"}

# Verify token
@app.post(API_VERSION + "validate")
def verify(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    scheme = credentials.scheme

    __check_create_admin_table()

    if __find_hashed_token(token) == None:
        raise HTTPException(
        status_code=401,
        detail="Unauthorized"
    )

    return True

@app.post(API_VERSION + "logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    scheme = credentials.scheme

    __check_create_admin_table()

    hashed_token = __find_hashed_token(token)

    if hashed_token == None:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE admins
                SET tokens = array_remove(tokens, %s)
                WHERE tokens @> ARRAY[%s]
            """, (hashed_token, hashed_token))

    return None

@app.post(API_VERSION + "log/submit")
def submit_log(
    user_id: UserID,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    __check_create_activitylog_table()

    timestamp_now = datetime.now(timezone.utc)

    print(user_id.user_id)
    print(timestamp_now)

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO activitylog (user_id, timestamp)
                VALUES (%s, %s)
                RETURNING *
            """, (user_id.user_id, timestamp_now))

            out = cur.fetchone()

            return {
                "event_id": out[0],
                "user_id": out[1],
                "timestamp": out[2]
            }

@app.get(API_VERSION + "log/{event_id}")
def get_one_log(
    event_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
    __check_create_activitylog_table()

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM activitylog
                WHERE event_id = %s
            """, (event_id,))

            out = cur.fetchone()

            if out == None:
                raise HTTPException(
                    status_code=404,
                    detail="User does not exist"
                )

            return {
                "event_id": out[0],
                "user_id": out[1],
                "timestamp": out[2]
            }

@app.post(API_VERSION + "log/")
def get_bulk_logs(
        params: LogBulkParams,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    limit = params.limit
    offset = params.offset

    if limit > 500:
        limit = 500

    __check_create_activitylog_table()

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM activitylog
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
             """, (limit, offset))

            all_logs = cur.fetchall()
            response = {}

            for event in all_logs:
                event_id = event[0]
                event_user = event[1]
                event_time = event[2]

                response[str(event_id)] = {
                    "event_id": event_id,
                    "user_id": event_user,
                    "timestamp": event_time
                }

            return response

# Insert a new user into the table
@app.post(API_VERSION + "enroll")
async def enroll_user(
    first_name: Annotated[str, Form()],
    last_name: Annotated[str, Form()],
    file: UploadFile,
    bio: Annotated[str, Form()] = "",
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    # Basic validation, done before anything is written.
    if file.content_type != "image/jpeg":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG."
        )

    __check_create_user_table()

    # Insert user into the table and retreive newly created ID.
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (first_name, last_name, bio, image_path)
                VALUES (%s, %s, %s, 'temp')
                RETURNING user_id
            """, (first_name, last_name, bio))

            user_id = cur.fetchone()[0]

    # Create directory path on disk
    dir_path = os.path.join(IMAGE_PATH, str(user_id))
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, "photo.jpg")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Update user record with the correct photo path after upload.
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users
                SET image_path = %s
                WHERE user_id = %s
            """, (file_path, user_id))

    # Tell the recognition service to load the new guest. If the photo has no
    # usable face, undo the enrollment so staff can retake the photo.
    recognition = await __recognition_sync(user_id, credentials)
    if recognition["status"] == "rejected":
        __delete_user_records(user_id)
        raise HTTPException(
            status_code=422,
            detail=recognition["detail"]
        )

    # Returns the newly created user as the response.
    user = get_user_by_id(user_id, credentials)
    user["recognition"] = recognition
    return user

# User deletion endpoint
@app.delete(API_VERSION + "users/{user_id}")
def delete_user(
    user_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    ext_headers = __unwrap_credentials(credentials)

    if not __delete_user_records(user_id):
        raise HTTPException(
            status_code=404,
            detail="User does not exist"
        )

    # Tell the recognition service to forget the guest. A 404 just means the
    # guest was never enrolled there, and if it's offline it won't load the
    # guest again on its next start anyway.
    try:
        httpx.delete(f"{RECOGNITION_URL}/faces/guests/{user_id}", headers=ext_headers, timeout=RECOGNITION_TIMEOUT)
    except httpx.HTTPError:
        pass

    return { "deleted_id": user_id }

# Should return a list of all users, which will be used by the machine learning server upon startup to inload users.
@app.get(API_VERSION + "users/")
def get_all_users(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:

            __check_create_user_table()

            cur.execute("""
                SELECT * FROM users
             """)

            all_users = cur.fetchall()
            response = {}

            for user in all_users:
                user_id = user[0]
                user_first_name = user[1]
                user_last_name = user[2]
                user_bio = user[3]
                user_image_path = user[4]

                response[str(user_id)] = {
                    "first_name": user_first_name,
                    "last_name": user_last_name,
                    "bio": user_bio,
                    "image_path": user_image_path
                }

            return response

# Fetches a single user based on their user ID.
@app.get(API_VERSION + "users/{user_id}")
def get_user_by_id(
    user_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:

            __check_create_user_table()

            cur.execute("""
                SELECT * FROM users WHERE user_id = %s
            """, (user_id,))

            user = cur.fetchone()

            if user == None:
                raise HTTPException(
                    status_code=404,
                    detail="User does not exist"
                )

            return {
                "id": user[0],
                "first_name": user[1],
                "last_name": user[2],
                "bio": user[3],
                "image_path": user[4]
            }

# Returns a user's photo, used by the staff app and the dashboard.
@app.get(API_VERSION + "users/{user_id}/photo")
def get_user_photo(
    user_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):

    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    user = get_user_by_id(user_id, credentials)
    if not user["image_path"] or not os.path.isfile(user["image_path"]):
        raise HTTPException(
            status_code=404,
            detail="User has no photo"
        )

    return FileResponse(user["image_path"], media_type="image/jpeg")

# Re-sends one user to the recognition service, e.g. if it was offline when
# they enrolled.
@app.post(API_VERSION + "users/{user_id}/sync")
async def sync_user(
    user_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    get_user_by_id(user_id, credentials)
    return await __recognition_sync(user_id, credentials)

# Whether the recognition service is running, and which user IDs it has loaded.
@app.get(API_VERSION + "recognition")
def get_recognition_status(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Does nothing if found or raises a 401 if Unauthorized
    verify(credentials)

    ext_headers = __unwrap_credentials(credentials)

    try:
        response = httpx.get(f"{RECOGNITION_URL}/faces/guests", headers=ext_headers, timeout=RECOGNITION_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError:
        return {"online": False, "enrolled": []}

    return {"online": True, "enrolled": sorted(int(i) for i in response.json())}

##
## HELPER METHODS, ETC
##

# Asks the recognition service to (re-)enroll a user from their saved photo.
# Returns {"status": "enrolled" | "rejected" | "offline", "detail": ...}.
# It's async because the recognition service calls back into this API to read
# the user, which would deadlock if this blocked the event loop.
async def __recognition_sync(user_id: int, credentials):
    verify(credentials)

    ext_headers = __unwrap_credentials(credentials)

    try:
        async with httpx.AsyncClient(timeout=RECOGNITION_TIMEOUT) as client:
            response = await client.post(f"{RECOGNITION_URL}/faces/guests/{user_id}/sync", headers=ext_headers)
    except httpx.HTTPError:
        return {
            "status": "offline",
            "detail": "The recognition service isn't running. The guest will be recognized after it starts."
        }

    if response.status_code == 422:
        return {"status": "rejected", "detail": response.json().get("detail", "No usable face in the photo.")}

    if response.is_error:
        return {"status": "offline", "detail": f"The recognition service returned an error ({response.status_code})."}

    return {"status": "enrolled", "detail": None}

# Deletes a user's row and photo folder. Returns False if the user didn't exist.
def __delete_user_records(user_id: int):
    __check_create_user_table()

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM users
                WHERE user_id = %s
                RETURNING user_id
            """, (user_id,))

            deleted = cur.fetchone() is not None

    shutil.rmtree(os.path.join(IMAGE_PATH, str(user_id)), ignore_errors=True)
    return deleted

# Creates the user table if it does not already exist.
def __check_create_user_table():
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'users'
                )
            """)

            result = cur.fetchone()

            if result[0]:
                return

            cur.execute("""
                CREATE TABLE users (
                    user_id SERIAL PRIMARY KEY,
                    first_name VARCHAR(30),
                    last_name  VARCHAR(30),
                    bio        TEXT,
                    image_path TEXT
                )
            """)

# Creates the admin table if it does not already exist.
def __check_create_admin_table():
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'admins'
                )
            """)

            result = cur.fetchone()

            if result[0]:
                return

            cur.execute("""
                CREATE TABLE admins (
                    admin_id SERIAL PRIMARY KEY,
                    username VARCHAR(16) UNIQUE,
                    passhash TEXT,
                    tokens TEXT[]
                )
            """)

            ph = PasswordHasher()
            passhash = ph.hash(INTERNAL_ADMIN_PW)

            # This is the default admin user used internally.
            cur.execute("""
                INSERT INTO admins (username, passhash)
                VALUES (%s, %s)
                RETURNING admin_id
            """, (INTERNAL_ADMIN_UN, passhash))

# Creates the ActivityLog table if it does not already exist.
def __check_create_activitylog_table():
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'activitylog'
                )
            """)

            result = cur.fetchone()

            if result[0]:
                return

            cur.execute("""
                CREATE TABLE activitylog (
                    event_id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    timestamp TIMESTAMPTZ
                )
            """)

# Given a raw token, find its stored hash in the table and return the hash or None.
def __find_hashed_token(token: str):
    tokens_all = None

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tokens FROM admins
            """)

            tokens_all = cur.fetchall()[0]

    ph = PasswordHasher()

    for user_tokens_hashed in tokens_all:
        for hashed_token in user_tokens_hashed:
            try:
                ph.verify(hashed_token, token)
                return hashed_token
            except:
                pass

    return None

def __unwrap_credentials(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    scheme = credentials.scheme

    header = { "Authorization": f"{scheme.capitalize()} {token}"}

    return header
