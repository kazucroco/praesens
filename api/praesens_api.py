from fastapi import FastAPI
import psycopg

app = FastAPI()

API_VERSION = "/api/v1/"
DB_CONNECTION_PARAMS = "host=localhost port=5432 dbname=praesens user=praesens password=praesens"

@app.get(API_VERSION)
def hello():
    return {"message": "Hello world!"}

# Should return a list of all users, which will be used by the machine learning server upon startup to inload users.
# NOTE: Right now, most of this code is not final and is just testing the API server.
@app.get(API_VERSION + "users/")
def get_all_users():
    with psycopg.connect(DB_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:

            __check_create_table()          
            
            # TODO: Remove this insertion. It's only to ensure that there is a value in the table for early testing.
            cur.execute("""
                INSERT INTO users (first_name, last_name, bio, photo_path) VALUES (
                    'Henry', 'Silva', 'test user', 'none'
                )
            """)
            
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

