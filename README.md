# Praesens
This repository is an active work in progress and is under active development.

## Database How-to
To run the PostgreSQL server, make sure you have Docker installed. Then run `docker compose up -d` in the project directory, and it should automatically pull and configure the PostgreSQL server for you.

## API How-to
These are quick and dirty instructions for getting the API server to run in your development environment. It expects the database server to be running and using the default parameters in `compose.yaml`.
1. Create a Python virtual environment along the lines of `python3 -m venv .venv` depending on your platform, **then activate it using** `source .venv/bin/activate` (Linux) or similar depending on your platform. 
2. Install `fastapi[standard]` and `psycopg[binary]` through `pip`.
3. Run `fastapi run api/praesens_api.py`

You can now make calls to the API server using [Postman](https://www.postman.com/) or whatever you wish.

## Staff Enrollment App
With the API and the recognition service running, open http://localhost:8000/ (it redirects to `/staff/`). Other devices on the same network can open `http://<this PC's IP>:8000/`.

- **Enroll guest** opens the enrollment form. The Photo frame shows the recognition service's camera, and **Capture** takes a still from it, so the webcam stays with the recognition service the whole time.
- **Guest list** shows every guest, whether recognition has loaded them, and a **Checkout** button that removes the guest and their face.
- **Guest popups:** when the camera recognizes an enrolled guest, a popup with their photo, name and notes appears in the top-right corner until it's dismissed. A "Possible match" popup has an amber edge.
- **Recognition:** after saving a guest, the API asks the recognition service to load them. If the photo has no usable face (or more than one), the guest isn't saved and staff are asked to retake the photo. If the recognition service isn't running, the guest is saved and loaded when it starts. Set `PRAESENS_RECOGNITION_URL` if it isn't at `http://localhost:8001`.
- **Camera:** the recognition service starts the camera when it starts. Set `PRAESENS_CAMERA_AUTOSTART=0` to start it by hand instead.

## Recognition How-to
The face recognition service lives in `recognition/` and runs alongside the API on port 8001. Install its dependencies with `pip install -r recognition/requirements.txt`, then run `fastapi run recognition/server.py --port 8001` while the API is running. It loads every user's photo from the API on startup. See `recognition/README.md` for endpoints, the dashboard WebSocket, and threshold tuning.
