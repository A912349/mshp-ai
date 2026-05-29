FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    fonts-dejavu-core \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2-mesa \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    gunicorn \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    wget -O /app/models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "6", "-k", "gthread", "--threads", "2", "-b", "0.0.0.0:5000", "app:app"]
