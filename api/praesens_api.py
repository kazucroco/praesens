import os
import shutil
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psycopg

app = FastAPI()

API_VERSION = "/api/v1/"
DB_CONNECTION_PARAMS = os.environ.get(
    "PRAESENS_DB",
    "host=localhost port=5432 dbname=praesens user=praesens password=praesens"
)

IMAGE_PATH = "./images/"

# The face recognition service (recognition/server.py). The API tells it about
# new and deleted guests so they're recognized (or forgotten) right away.
RECOGNITION_URL = os.environ.get("PRAESENS_RECOGNITION_URL", "http://localhost:8001")
RECOGNITION_TIMEOUT = 30.0

# The staff enrollment app (api/staff/) is served at /staff/.
STAFF_APP_PATH = Path(__file__).parent / "staff"

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/staff/")

@app.get(API_VERSION)
def hello():
    return {"message": "Hello world!"}

# Insert a new user into the table
@app.post(API_VERSION + "enroll/")
async def enroll_user(
    first_name: Annotated[str, Form()],
    last_name: Annotated[str, Form()],
    file: UploadFile,
    bio: Annotated[str, Form()] = ""):

    # Basic validation, done before anything is written.
    if file.content_type != "image/jpeg":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG."
        )

    __check_create_table()

    # Insert user into the table and retreive newly created ID.
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (first_name, last_name, bio, photo_path)
                VALUES (%s, %s, %s, 'temp')
                RETURNING id
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
                SET photo_path = %s
                WHERE id = %s
            """, (file_path, user_id))

    # Tell the recognition service to load the new guest. If the photo has no
    # usable face, undo the enrollment so staff can retake the photo.
    recognition = await __recognition_sync(user_id)
    if recognition["status"] == "rejected":
        __delete_user_records(user_id)
        raise HTTPException(
            status_code=422,
            detail=recognition["detail"]
        )

    # Returns the newly created user as the response.
    user = get_user_by_id(user_id)
    user["recognition"] = recognition
    return user

# User deletion endpoint
@app.delete(API_VERSION + "users/{user_id}")
def delete_user(user_id: int):
    if not __delete_user_records(user_id):
        raise HTTPException(
            status_code=404,
            detail="User does not exist"
        )

    # Tell the recognition service to forget the guest. A 404 just means the
    # guest was never enrolled there, and if it's offline it won't load the
    # guest again on its next start anyway.
    try:
        httpx.delete(f"{RECOGNITION_URL}/faces/guests/{user_id}", timeout=RECOGNITION_TIMEOUT)
    except httpx.HTTPError:
        pass

    return user_id

# Should return a list of all users, which will be used by the machine learning server upon startup to inload users.
@app.get(API_VERSION + "users/")
def get_all_users():
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:

            __check_create_table()

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

                response[user_id] = {
                    "first_name": user_first_name,
                    "last_name": user_last_name,
                    "bio": user_bio,
                    "image_path": user_image_path
                }

            return response

# Fetches a single user based on their user ID.
@app.get(API_VERSION + "users/{user_id}")
def get_user_by_id(user_id: int):
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:

            __check_create_table()

            cur.execute("""
                SELECT * FROM users WHERE id = %s
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
def get_user_photo(user_id: int):
    user = get_user_by_id(user_id)
    if not user["image_path"] or not os.path.isfile(user["image_path"]):
        raise HTTPException(
            status_code=404,
            detail="User has no photo"
        )

    return FileResponse(user["image_path"], media_type="image/jpeg")

# Re-sends one user to the recognition service, e.g. if it was offline when
# they enrolled.
@app.post(API_VERSION + "users/{user_id}/sync")
async def sync_user(user_id: int):
    get_user_by_id(user_id)
    return await __recognition_sync(user_id)

# Whether the recognition service is running, and which user IDs it has loaded.
@app.get(API_VERSION + "recognition/")
def get_recognition_status():
    try:
        response = httpx.get(f"{RECOGNITION_URL}/faces/guests", timeout=RECOGNITION_TIMEOUT)
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
async def __recognition_sync(user_id: int):
    try:
        async with httpx.AsyncClient(timeout=RECOGNITION_TIMEOUT) as client:
            response = await client.post(f"{RECOGNITION_URL}/faces/guests/{user_id}/sync")
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
    __check_create_table()

    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM users
                WHERE id = %s
                RETURNING id
            """, (user_id,))

            deleted = cur.fetchone() is not None

    shutil.rmtree(os.path.join(IMAGE_PATH, str(user_id)), ignore_errors=True)
    return deleted

# Creates the user table if it does not already exist.
def __check_create_table():
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
                    id SERIAL PRIMARY KEY,
                    first_name VARCHAR(30),
                    last_name  VARCHAR(30),
                    bio        TEXT,
                    photo_path TEXT
                )
            """)

# Mounted last so it doesn't shadow the API routes above.
app.mount("/staff", StaticFiles(directory=STAFF_APP_PATH, html=True), name="staff")
