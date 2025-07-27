# Dockerfile


FROM python:3.9-slim-bullseye


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# 7. Define the command to run the application
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]