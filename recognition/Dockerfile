FROM python:3.14
WORKDIR /recognition

# Required install for cv2
RUN apt update && apt install -y libgl1

# Copy server files to container.
COPY requirements.txt .
COPY praesens_face ./praesens_face
COPY server.py .

EXPOSE 8001

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt 

CMD ["fastapi", "run", "server.py", "--port", "8001"]
