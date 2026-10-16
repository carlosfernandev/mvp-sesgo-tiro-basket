# WHY python:3.11-slim and not an Alpine base: mediapipe and
# opencv-contrib-python publish manylinux (glibc) wheels only, so an Alpine
# musl image would force a from-source OpenCV build. The slim Debian base
# keeps the image small while using the prebuilt wheels.
FROM python:3.11-slim

# WHY these system packages: ffmpeg is invoked by the video pipeline, and
# importing cv2/mediapipe needs libgl1 + libglib2.0-0 even in a headless
# container (a missing libglib.2.0 raises ImportError at `import cv2`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# WHY requirements copied first: this layer is cached and only rebuilt when
# the dependency set changes, not on every source edit.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests

# WHY FLASK_HOST=0.0.0.0: the dev default binds 127.0.0.1, which is
# unreachable from outside the container, so the published port would never
# answer. The path env vars point at the mounted volumes below instead of the
# CWD-anchored fallbacks.
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5001 \
    MODEL_PATH=/app/models/pose_landmarker_lite.task \
    DATA_RAW_DIR=/data/raw \
    DATA_PROCESSED_DIR=/data/processed \
    ANALISIS_RESULTS_DIR=/data/processed

EXPOSE 5001

# WHY the module path: app.py's __main__ block bootstraps sys.path for direct
# execution, and the analysis core is imported lazily after that bootstrap, so
# running the file directly is CWD-independent.
CMD ["python", "src/api/app.py"]
