import os
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
import psycopg

app = FastAPI()

API_VERSION = "/api/v1/"
DB_CONNECTION_PARAMS = "host=localhost port=5432 dbname=praesens user=praesens password=praesens"

IMAGE_PATH = "./images/"

@app.get(API_VERSION)
def hello():
    return {"message": "Hello world!"}

# Insert a new user into the table
@app.post(API_VERSION + "enroll/")
async def enroll_user(
    first_name: Annotated[str, Form()],
    last_name: Annotated[str, Form()],
    bio: Annotated[str, Form()],
    file: UploadFile):

    __check_create_table()

    # Placeholder value which gets updated by the database cursor.
    user_id = -1

    # Insert user into the table and retreive newly created ID.
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO users (first_name, last_name, bio, photo_path)
                VALUES ('{first_name}', '{last_name}', '{bio}', 'temp')
                RETURNING id
            """)

            user_id = cur.fetchone()[0]

    # Basic validation.
    if file.content_type != "image/png":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PNG."
        )
    
    # Create directory path on disk
    dir_path = os.path.join(IMAGE_PATH, str(user_id))
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, "photo.png")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Update user record with the correct photo path after upload.
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE users
                SET photo_path = '{file_path}'
                WHERE id = {user_id}
            """)

    # Returns the newly created user as the response.
    return get_user_by_id(user_id)

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

            # NOTE: This is probably extremely vulnerable to SQL injection.
            cur.execute(f"""
                SELECT * FROM users WHERE id = {user_id}
            """)

            user = cur.fetchone()

            return {
                "id": user[0],
                "first_name": user[1],
                "last_name": user[2],
                "bio": user[3],
                "image_path": user[4]
            }

##
## HELPER METHODS, ETC
##

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