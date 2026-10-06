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

> NOTE: The API server currently expects HTTP (**NOT** HTTPS) requests. Apparently, it's possible to get this working with FastAPI directly but I think I'm probably just going to set up a Caddy server in the `compose.yaml` because it seems easier. I just haven't done it yet.