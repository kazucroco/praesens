FROM python:3.14
WORKDIR /database_api

# Copy server files to container.
COPY requirements.txt . 
COPY database_api.py .

EXPOSE 8000

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt 

CMD ["fastapi", "run", "database_api.py"]
