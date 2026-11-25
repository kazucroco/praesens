import requests
import os

class TestDatabaseApi:
    # Setup:
    # Attempts to use https://hostname.local/api/v1 but will fall back to
    # configurable host and port from environment variables.
    hostname = os.environ.get("HOSTNAME", None)
    host     = "localhost"
    if hostname:
        host = hostname + ".local"
    port     = os.environ.get("TEST_DB_API_PORT", 8443)
    url_base = os.environ.get(
        "TEST_DB_API_URL",
        f"https://{host}:{str(port)}/api/v1"
    )

    #
    # TESTS
    #

    # Hello world / "is online" endpoint
    def test_hello(self):
        response = requests.get(self.url_base + "/", verify=False)
        assert response.status_code == 200

    #
    # LOGIN
    #

    # Login (and technically validation, logout)
    def test_login_good(self):
        # Tests login and validate against received token.
        header = self._get_auth_header()
        assert self._validate_header(header) == True

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Tests invalid credentials
    def test_login_bad(self):
        body = {
            "username": "", "password": "" }
        response = requests.post(self.url_base + "/login", json=body, verify=False)
        assert response.status_code == 401

    # Tests username too long
    def test_login_bad_long(self):
        body = {
            "username": "morethanthirtycharacterusername", "password": "yes" }
        response = requests.post(self.url_base + "/login", json=body, verify=False)
        assert response.status_code == 400

    # Tests bad body schema
    def test_login_bad_fmt(self):
        body = { "foo": "bar" }
        response = requests.post(self.url_base + "/login", json=body, verify=False)
        assert response.status_code == 422

    #
    # VALIDATE
    #

    # Login validation when not logged in
    def test_validate_unauth(self):
        response = requests.post(self.url_base + "/validate", verify=False)
        assert response.status_code == 401

    # Logout when not logged in
    def test_logout_unauth(self):
        response = requests.post(self.url_base + "/logout", verify=False)
        assert response.status_code == 401

    #
    # ACTIVITY LOG
    #

    # Submit activity log entry
    def test_log_submit_good(self):
        header = self._get_auth_header()

        # Since there's no endpoint to delete items from the log, I'll treat
        # negative numbers as test values and hide from the frontend.
        body = { "user_id": -1 }

        response = requests.post(
            self.url_base + "/log/submit",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 200

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Submit activity log entry bad format
    def test_log_submit_bad_fmt(self):
        header = self._get_auth_header()

        # Since there's no endpoint to delete items from the log, I'll treat
        # negative numbers as test values and hide from the frontend.
        body = { "foo": -1 }

        response = requests.post(
            self.url_base + "/log/submit",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 422

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Submit activity log entry unauth
    def test_log_submit_bad_unauth(self):
        body = { "user_id": -1 }

        response = requests.post(
            self.url_base + "/log/submit",
            json=body,
            verify=False
        )
        assert response.status_code == 401

    # Fetch specific log id
    def test_log_fetchone_good(self):
        header = self._get_auth_header()
        body = { "user_id": -1 }

        response = requests.post(
            self.url_base + "/log/submit",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 200

        json = response.json()
        event_id = json["event_id"]

        response = requests.get(
            self.url_base + f"/log/{event_id}",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 200

        json = response.json()
        assert json["user_id"] == -1
        assert json["event_id"] == event_id

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Fetch specific log nonexistent
    def test_log_fetchone_dne(self):
        header = self._get_auth_header()
        body = { "user_id": -1 }

        event_id = -1

        response = requests.get(
            self.url_base + f"/log/{event_id}",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 404

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Fetch specific log unauth
    def test_log_fetchone_unauth(self):
        body = { "user_id": -1 }

        event_id = 1

        response = requests.get(
            self.url_base + f"/log/{event_id}",
            json=body,
            verify=False
        )
        assert response.status_code == 401

    # Fetch bulk logs good, no params
    def test_log_fetchbulk_good_noparams(self):
        header = self._get_auth_header()

        body = {}

        response = requests.post(
            self.url_base + "/log/",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 200

        # Log fetch limit is 500
        json = response.json()
        assert len(json) <= 500

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Fetch bulk logs good, limit
    def test_log_fetchbulk_good_limit(self):
        header = self._get_auth_header()

        body = { "limit": 1 }

        response = requests.post(
            self.url_base + "/log/",
            json=body,
            headers=header,
            verify=False
        )
        assert response.status_code == 200

        json = response.json()
        assert len(json) <= 1

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Fetch bulk logs good, testing offset
    def test_log_fetchbulk_good_offset(self):
        header = self._get_auth_header()

        body1 = { "limit": 2, "offset": 0 }
        body2 = { "limit": 1, "offset": 1 }

        response1 = requests.post(
            self.url_base + "/log/",
            json=body1,
            headers=header,
            verify=False
        )
        assert response1.status_code == 200

        response2 = requests.post(
            self.url_base + "/log/",
            json=body2,
            headers=header,
            verify=False
        )
        assert response2.status_code == 200

        json1 = response1.json()
        json2 = response2.json()

        # R1: { "foo": {}, "bar": {} }
        #                  ^^^^^^^^^
        # R2: { "bar": {} }
        #       ^^^^^^^^^
        # Should be the same with correctly-working offsets.
        r1_second_log = list(json1.items())[1]
        r2_first_log  = list(json2.items())[0]
        assert r1_second_log == r2_first_log

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    #
    # ENROLLMENT
    #

    # Good enrollment
    def test_enroll_good(self):
        header = self._get_auth_header()

        # Adds a valid user
        response = None
        with open("tests/test_images/1.jpeg", "rb") as image:
            data  = {
                "first_name": "Test",
                "last_name" : "User1",
                "bio"       : "The first test user."
            }
            files = {
                "file": ("1.jpeg", image, "image/jpeg")
            }

            response = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                headers=header,
                verify=False
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        json = response.json()

        assert json["first_name"] == "Test"
        assert json["last_name"]  == "User1"
        assert json["bio"]        == "The first test user."

        added_user_id = json["id"]

        # Delete the newly added user
        response = requests.delete(self.url_base + f"/users/{str(added_user_id)}", headers=header, verify=False)
        assert response.status_code == 200

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Unauthorized enrollment
    def test_enroll_unauth(self):
        response = None
        with open("tests/test_images/1.jpeg", "rb") as image:
            data  = {
                "first_name": "Test",
                "last_name" : "Unauthorized",
                "bio"       : "The first test user."
            }
            files = {
                "file": ("1.jpeg", image, "image/jpeg")
            }

            response = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                verify=False
            )
        assert response.status_code == 401

    # Unprocessable if any fields are missing.
    def test_enroll_bad_missingdata(self):
        header = self._get_auth_header()
        data  = {
                "first_name": "Photoless",
                "last_name" : "User",
                "bio"       : "This user does not have a photo."
            }

        response = requests.post(
            self.url_base + "/enroll",
            data=data,
            headers=header,
            verify=False)
        assert response.status_code == 422

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Request should fail if no face is detected
    def test_enroll_bad_noface(self):
        header = self._get_auth_header()

        # Tries to upload a photo of a buger
        response = None
        with open("tests/test_images/3.jpeg", "rb") as image:
            data  = {
                "first_name": "Cheese",
                "last_name" : "Burger",
                "bio"       : "It is not a human."
            }
            files = {
                "file": ("3.jpeg", image, "image/jpeg")
            }

            response = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                headers=header,
                verify=False
            )
        assert response.status_code == 422

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    #
    # DELETION
    # - Successful deletion was tested in the user add, so it's omitted here.
    #

    # Deletes a nonexistent user
    def test_delete_dne(self):
        header = self._get_auth_header()

        response = requests.delete(self.url_base + "/users/-1", headers=header, verify=False)
        assert response.status_code == 404

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Unauthorized delete
    def test_delete_unauth(self):
        response = requests.delete(self.url_base + "/users/-1", verify=False)
        assert response.status_code == 401

    #
    # GET SINGLE USER
    #

    # Get single user good
    def test_get_user_good(self):
        header = self._get_auth_header()

        # Adds a sample user
        response1 = None
        with open("tests/test_images/2.jpeg", "rb") as image:
            data  = {
                "first_name": "Test",
                "last_name" : "User2",
                "bio"       : "The second test user."
            }
            files = {
                "file": ("2.jpeg", image, "image/jpeg")
            }
            response1 = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                headers=header,
                verify=False
            )
        assert response1.status_code == 200
        json1 = response1.json()

        added_user_id = json1["id"]

        response2 = requests.get(
            self.url_base + f"/users/{added_user_id}",
                headers=header,
                verify=False
        )
        json2 = response2.json()

        assert json1["id"] == json2["id"]
        assert json1["first_name"] == json2["first_name"]
        assert json1["last_name"] == json2["last_name"]
        assert json1["bio"] == json2["bio"]
        assert json1["image_path"] == json2["image_path"]

        # Clean up
        self._del_user(added_user_id)

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Get a nonexistent user
    def test_get_user_dne(self):
        header = self._get_auth_header()

        response = requests.get(self.url_base + "/users/-1", headers=header, verify=False)
        assert response.status_code == 404

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Unauthorized get
    def test_get_user_unauth(self):
        response = requests.get(self.url_base + "/users/-1", verify=False)
        assert response.status_code == 401

    #
    # GET ALL USERS
    #

    # Attempts successful get of all users
    def test_get_all_users_good(self):
        header = self._get_auth_header()

        added_user_id = self._add_testuser_1()

        # Gets all users and tests if the new user is in the list
        response = requests.get(self.url_base + "/users/", headers=header, verify=False)
        assert response.status_code == 200
        json = response.json()

        assert str(added_user_id) in json

        # Delete the newly added user and verify that they are no longer in the list
        self._del_user(added_user_id)

        response = requests.get(self.url_base + "/users/", headers=header, verify=False)
        assert response.status_code == 200
        json = response.json()

        assert str(added_user_id) not in json

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Unauthorized get all users
    def test_get_all_users_unauth(self):
        response = requests.get(self.url_base + "/users/", verify=False)
        assert response.status_code == 401

    #
    # SINGLE USER SYNC
    #
    def test_sync_good(self):
        header = self._get_auth_header()

        # Adds a valid user
        added_user_id = self._add_testuser_1()

        response = requests.post(self.url_base + f"/users/{str(added_user_id)}/sync", headers=header, verify=False)
        assert response.status_code == 200

        # Delete the newly added user and verify that they are no longer in the list
        self._del_user(added_user_id)

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Attempt to sync a nonexistent user
    def test_sync_dne(self):
        header = self._get_auth_header()

        response = requests.post(self.url_base + f"/users/-1/sync", headers=header, verify=False)
        assert response.status_code == 404

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Attempt to sync when unauthorized
    def test_sync_unauth(self):
        response = requests.post(self.url_base + f"/users/-1/sync", verify=False)
        assert response.status_code == 401

    #
    # RECOGNITION SERVER STATUS
    #

    # Authorized status check
    def test_recognition_good(self):
        header = self._get_auth_header()

        response = requests.get(self.url_base + "/recognition", headers=header, verify=False)
        assert response.status_code == 200

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Unauthorized status check
    def test_recognition_bad(self):
        response = requests.get(self.url_base + "/recognition", verify=False)
        assert response.status_code == 401

    #
    # USER PHOTO
    #

    # Tests the fetched image with the original copy
    def test_user_photo_good(self):
        header = self._get_auth_header()

        # Adds a sample user
        response1 = None
        with open("tests/test_images/2.jpeg", "rb") as image:
            image_bytes = image.read()

            data  = {
                "first_name": "Test",
                "last_name" : "User2",
                "bio"       : "The second test user."
            }
            files = {
                "file": ("2.jpeg", image_bytes, "image/jpeg")
            }
            response1 = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                headers=header,
                verify=False
            )

            json1 = response1.json()
            added_user_id = json1["id"]

            response2 = requests.get(self.url_base + f"/users/{added_user_id}/photo", headers=header, verify=False)
            assert response2.content == image_bytes

            self._del_user(added_user_id)

        assert response1.status_code == 200

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Compares the returned photo with a different photo
    def test_user_photo_different(self):
        header = self._get_auth_header()

        # Adds a sample user
        response1 = None
        with open("tests/test_images/2.jpeg", "rb") as image:
            image_bytes = image.read()
            image2_bytes = open("tests/test_images/3.jpeg", "rb").read()

            data  = {
                "first_name": "Test",
                "last_name" : "User2",
                "bio"       : "The second test user."
            }
            files = {
                "file": ("2.jpeg", image_bytes, "image/jpeg")
            }
            response1 = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                headers=header,
                verify=False
            )

            json1 = response1.json()
            added_user_id = json1["id"]

            response2 = requests.get(self.url_base + f"/users/{added_user_id}/photo", headers=header, verify=False)
            assert response2.content != image2_bytes

            self._del_user(added_user_id)

        assert response1.status_code == 200

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

    # Makes an unauthorized call to the image photo endpoint
    def test_user_photo_unauth(self):
        added_user_id = self._add_testuser_1()

        response = requests.get(self.url_base + f"/users/{added_user_id}/photo", verify=False)
        assert response.status_code == 401

        self._del_user(added_user_id)

    #
    # HELPERS
    #

    # Runs login and returns the authorization header. Since most endpoints
    # require authorization this is reused a lot.
    def _get_auth_header(self):
        username = os.environ.get("INTERNAL_ADMIN_UN", "praesens")
        password = os.environ.get("INTERNAL_ADMIN_PW", "changeme")

        body = { "username": username, "password": password }

        # Tests login
        response = requests.post(self.url_base + "/login", json=body, verify=False)
        assert response.status_code == 200

        # Tests generated token
        json = response.json()
        header = {
            "Authorization": f"{json["token_type"].capitalize()} {json["access_token"]}"
        }
        return header

    def _validate_header(self, header: dict):
        response = requests.post(self.url_base + "/validate", headers=header, verify = False)
        return response.ok

    # Since several endpoints require user operations, this adds a test user.
    def _add_testuser_1(self):
        header = self._get_auth_header()

        # Adds a valid user
        response = None
        with open("tests/test_images/1.jpeg", "rb") as image:
            data  = {
                "first_name": "Test",
                "last_name" : "User1",
                "bio"       : "The first test user."
            }
            files = {
                "file": ("1.jpeg", image, "image/jpeg")
            }

            response = requests.post(
                self.url_base + "/enroll",
                data=data,
                files=files,
                headers=header,
                verify=False
            )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        json = response.json()

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200

        added_user_id = json["id"]
        return added_user_id

    def _del_user(self, user_id: int):
        header = self._get_auth_header()

        # Delete the newly added user
        response = requests.delete(self.url_base + f"/users/{str(user_id)}", headers=header, verify=False)
        assert response.status_code == 200

        # Deletes generated token from database
        response = requests.post(self.url_base + "/logout", headers=header, verify=False)
        assert response.status_code == 200
